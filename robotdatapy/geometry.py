from dataclasses import dataclass
import numpy as np

@dataclass
class Circle:
    x: float
    y: float
    radius: float
    
    @property
    def center(self):
        return np.array([self.x, self.y])

def circle_intersection(
    circle1: Circle = None,
    circle2: Circle = None,
    center1: np.array = None,
    center2: np.array = None,
    radius1: float = None,
    radius2: float = None,
):
    """Calculates circle intersection area. Function should be called either with circle objects 
    or centers and radii.

    Args:
        circle1 (Circle, optional): First circle. Defaults to None.
        circle2 (Circle, optional): Second circle. Defaults to None.
        center1 (np.array, optional): Center of first circle. Defaults to None.
        center2 (np.array, optional): Center of second circle. Defaults to None.
        radius1 (float, optional): Radius of first circle. Defaults to None.
        radius2 (float, optional): Radius of second circle. Defaults to None.
    """
    assert (circle1 is not None and circle2 is not None) or \
        (center1 is not None and center2 is not None and radius1 is not None and radius2 is not None), \
        "Either provide circle objects or centers and radii"
    assert (circle1 is None and circle2 is None) or \
        (center1 is None and center2 is None and radius1 is None and radius2 is None), \
        "Provide either circle objects or centers and radii"
        
    if circle1 is None and circle2 is None:
        circle1 = Circle(center1.item(0), center1.item(1), radius1)
        circle2 = Circle(center2.item(0), center2.item(1), radius2)
        
    d = np.linalg.norm(circle1.center - circle2.center)
    if d <= np.abs(circle1.radius - circle2.radius): # one circle is inside the other
        return np.pi * np.min([circle1.radius, circle2.radius])**2
    elif d >= circle1.radius + circle2.radius: # circles do not intersect
        return 0.
    r1 = circle1.radius
    r2 = circle2.radius
    return (r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1)) \
        + r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2)) \
        - 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)))
