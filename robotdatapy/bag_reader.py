"""
Fast raw-message iteration for rosbags' AnyReader.

rosbags reads MCAP files one whole chunk at a time: for every chunk containing
a requested topic, the entire chunk is read from disk and scanned for matching
records. When a small topic (odometry, tf) is recorded alongside large ones
(uncompressed images), nearly every chunk contains the small topic, so reading
it means reading essentially the whole bag.

MCAP stores a per-channel MessageIndex after each chunk giving the log time and
offset of every message in that chunk. For uncompressed chunks, iter_raw_messages
uses these indexes to read only the requested messages (with many positional
reads in flight at once) and skips everything else.

The fast path covers ROS2 MCAP storage with uncompressed chunks. Everything else
(ROS1 bags, sqlite3 storage, compressed chunks/messages/files, unindexed files,
untested rosbags versions) falls back to AnyReader.messages. Either way, the
yielded messages, their order, and their bytes are identical to
AnyReader.messages.

The fast path relies on rosbags internals (McapReader.chunks/.channels/.path,
DirectoryReader.storages), so it is gated to rosbags versions it was tested
against.
"""

import heapq
import itertools
import logging
import os
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from itertools import groupby

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.rosbag2.reader import DirectoryReader
from rosbags.rosbag2.storage_mcap import McapReader

logger = logging.getLogger(__name__)

_TESTED_ROSBAGS_VERSIONS = ("0.11.",)

_MCAP_OP_MESSAGE = 0x05
_MCAP_OP_MESSAGE_INDEX = 0x07
# Chunk record: opcode (1) + record length (8) + message_start_time (8) +
# message_end_time (8) + uncompressed_size (8) + uncompressed_crc (4) +
# compression string length (4) + compression string + records length (8)
_MCAP_CHUNK_HEADER_FIXED_LEN = 9 + 40

# Small reads scattered across a large file are latency-bound when issued one at
# a time (especially on USB SSDs); keeping many in flight lets the drive serve
# them in parallel. Threads release the GIL in os.pread.
_READ_WORKERS = 16
_READ_AHEAD = 256
# First read per message record; larger records get a second read for the rest.
_FIRST_READ_BYTES = 4096

_warned = set()


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg)


def _fast_path_supported():
    rosbags_version = version("rosbags")
    if rosbags_version.startswith(_TESTED_ROSBAGS_VERSIONS):
        return True
    _warn_once(
        "version",
        f"robotdatapy fast bag reading is untested with rosbags {rosbags_version}; "
        "falling back to standard (slower) reading.",
    )
    return False


def iter_raw_messages(reader: AnyReader, connections, start=None, stop=None):
    """
    Drop-in replacement for AnyReader.messages(connections=..., start=..., stop=...).

    Args:
        reader (AnyReader): An open AnyReader.
        connections (list): Connections to read (from reader.connections). An empty
            list means all connections, as with AnyReader.messages.
        start (int, optional): Yield only messages at or after this bag time (ns).
        stop (int, optional): Yield only messages before this bag time (ns).

    Yields:
        tuple: (connection, bag timestamp in ns, raw serialized message bytes)
    """
    connections = list(connections)
    if not connections or not _fast_path_supported():
        yield from reader.messages(connections=connections, start=start, stop=stop)
        return

    # Mirrors AnyReader.messages: one generator per underlying bag reader, merged
    # by timestamp.
    generators = [
        _reader_messages(owner, list(conns), start, stop)
        for owner, conns in groupby(
            sorted(connections, key=lambda x: id(x.owner)), key=lambda x: x.owner
        )
    ]
    yield from heapq.merge(*generators, key=lambda x: x[1])


def _reader_messages(owner, connections, start, stop):
    """
    Messages from the reader that owns these connections (what AnyReader.messages
    calls): a DirectoryReader for rosbag2 directories, a McapReader for a bare
    .mcap file, or a rosbag1 Reader.
    """
    if isinstance(owner, McapReader):
        yield from _storage_messages(owner, connections, start, stop)
    elif isinstance(owner, DirectoryReader) and not owner.metadata.compression_mode:
        # Mirrors DirectoryReader.messages: storage files in order, each file's
        # connections mapped back to the directory-level connections.
        topics = [x.topic for x in connections]
        for sub in owner.storages:
            sub_conns = [x for x in sub.connections if x.topic in topics]
            connmap = {
                x.id: next(y for y in connections if x.topic == y.topic)
                for x in sub_conns
            }
            for sub_conn, timestamp, data in _storage_messages(
                sub, sub_conns, start, stop
            ):
                yield connmap[sub_conn.id], timestamp, data
    else:
        # ROS1 bag, bare sqlite3 file, or message/file-compressed rosbag2 directory
        yield from owner.messages(connections=connections, start=start, stop=stop)


def _storage_messages(storage, connections, start, stop):
    """Messages from one storage file, via MCAP message indexes when possible."""
    if isinstance(storage, McapReader):
        fd = os.open(storage.path, os.O_RDONLY)
        try:
            with ThreadPoolExecutor(_READ_WORKERS) as pool:
                try:
                    entries = _mcap_index_entries(
                        storage, connections, start, stop, fd, pool
                    )
                except Exception as e:
                    _warn_once(
                        ("index", str(storage.path)),
                        f"Could not use MCAP message indexes for {storage.path} "
                        f"({e!r}); falling back to standard (slower) reading.",
                    )
                    entries = None
                if entries is not None:
                    yield from _read_mcap_entries(storage, entries, fd, pool)
                    return
        finally:
            os.close(fd)
    yield from storage.messages(connections, start, stop)


def _mcap_index_entries(storage: McapReader, connections, start, stop, fd, pool):
    """
    Collect (log_time, order_offset, file_offset, channel_id, connection) for every
    requested message, sorted in the same order McapReader.messages yields them.
    Returns None if this file cannot use the fast path.
    """
    if not storage.chunks:
        return None  # unchunked file; McapReader scans it linearly

    # Same channel matching as McapReader.messages
    channel_map = {
        cid: conn
        for conn in connections
        if (
            cid := next(
                (
                    cid
                    for cid, x in storage.channels.items()
                    if x.schema == conn.msgtype and x.topic == conn.topic
                ),
                None,
            )
        )
        is not None
    }

    # Same chunk selection as McapReader.messages
    chunks = [
        x
        for x in storage.chunks
        if (start is None or start < x.message_end_time)
        and (stop is None or x.message_start_time < stop)
        and (any(x.channel_count.get(cid, 0) for cid in channel_map))
    ]
    if any(
        chunk.compression
        or any(
            chunk.channel_count.get(cid, 0) and cid not in chunk.message_index_offsets
            for cid in channel_map
        )
        for chunk in chunks
    ):
        return None

    # One MessageIndex record per (chunk, channel): opcode (1) + record length (8)
    # + channel_id (2) + entries byte length (4) + (log_time, offset) u64 pairs
    jobs = [
        (chunk, cid, conn)
        for chunk in chunks
        for cid, conn in channel_map.items()
        if chunk.channel_count.get(cid, 0)
    ]

    def read_index(job):
        chunk, cid, _ = job
        size = 15 + 16 * chunk.channel_count[cid]
        return os.pread(fd, size, chunk.message_index_offsets[cid])

    entries = []
    for (chunk, cid, conn), record in zip(jobs, pool.map(read_index, jobs)):
        op, _, channel_id, entries_len = struct.unpack_from("<BQHI", record, 0)
        if op != _MCAP_OP_MESSAGE_INDEX or channel_id != cid:
            raise ValueError(
                f"unexpected MessageIndex record (opcode {op:#x}, channel {channel_id})"
            )
        if 15 + entries_len != len(record):
            raise ValueError("MessageIndex length does not match channel count")
        index = np.frombuffer(record, dtype="<u8", offset=15).reshape(-1, 2)
        lo = start or chunk.message_start_time
        hi = stop or chunk.message_end_time + 1
        records_start = (
            chunk.chunk_start_offset
            + _MCAP_CHUNK_HEADER_FIXED_LEN
            + len(chunk.compression)
        )
        for log_time, offset in index.tolist():
            if lo <= log_time < hi:
                # McapReader orders by (log_time, chunk_start_offset + offset)
                entries.append(
                    (
                        log_time,
                        chunk.chunk_start_offset + offset,
                        records_start + offset,
                        cid,
                        conn,
                    )
                )
    entries.sort(key=lambda x: (x[0], x[1]))
    return entries


def _read_mcap_entries(storage: McapReader, entries, fd, pool):
    """
    Read and yield each indexed message record in order, validating it against its
    index. Keeps at most _READ_AHEAD reads in flight so memory stays bounded even
    for large (image) messages.
    """

    def read_record(entry):
        log_time, _, file_offset, cid, _ = entry
        data = os.pread(fd, _FIRST_READ_BYTES, file_offset)
        op, length = struct.unpack_from("<BQ", data, 0)
        if 9 + length > len(data):
            data += os.pread(fd, 9 + length - len(data), file_offset + len(data))
        channel_id, _, record_log_time, _ = struct.unpack_from("<HIQQ", data, 9)
        if (
            op != _MCAP_OP_MESSAGE
            or channel_id != cid
            or record_log_time != log_time
            or len(data) < 9 + length
        ):
            raise RuntimeError(
                f"MCAP message index mismatch in {storage.path} at offset {file_offset}"
            )
        return data[9 + 22 : 9 + length]

    in_flight = deque()
    entries = iter(entries)
    for entry in itertools.islice(entries, _READ_AHEAD):
        in_flight.append((entry, pool.submit(read_record, entry)))
    while in_flight:
        (log_time, _, _, _, conn), future = in_flight.popleft()
        for entry in itertools.islice(entries, 1):
            in_flight.append((entry, pool.submit(read_record, entry)))
        yield conn, log_time, future.result()
