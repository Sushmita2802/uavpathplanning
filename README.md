# uavpathplanning
MATLAB implementation of obstacle-aware UAV navigation. Includes building-like obstacles, path planning with 3D A* (26-connectivity), shortcut and spline smoothing, and animated UAV trajectory visualization.
This project implements **3D UAV path planning** using an **A\*** search algorithm with:
- Obstacle representation as 3D voxel grids (buildings).
- Path planning with **A\*** (26-neighbor connectivity).
- **Shortcut smoothing** (line-of-sight optimization).
- **Spline smoothing** for a safe, continuous trajectory.
- **Animated UAV visualization** (quadcopter model following the planned path).
- Buildings displayed as **colored, semi-transparent blocks**.
