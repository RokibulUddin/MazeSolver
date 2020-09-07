# A* MazeSolver

This is a simple maze creator/solver and solution visualizer using A* algorithm
![alt text](https://raw.githubusercontent.com/RokibulUddin/MazeSolver/master/example.png)

## Usage:
```bash
python[.exe] MazeSolver.py [maze_input_<num>.png]
```
if required, install required modules with:
```bash
pip install -r requirements.txt
```

1) Execute it to generate a random maze
2) Click any point in the maze to create a **start point**
3) Click any point in the maze to create a **stop point**
4) Press **SPACE** button to start the solver by step by step
    1) Press **f** button to toggle between quick or step-by-step solution
    2) Once started a step-by-step solution is not possible to press **f**, you have to wait
    3) So, press **f** and then **SPACE** to have quick solution
    4) If you regenerate the maze, there is no need to re-press **f**
* Press **s** to save the maze (once solved)
    * it will create 3 images:
        1) maze_<num>.png = the maze with start and end point
        2) maze_sol_<num>.png = the maze with the solution on it
        3) maze_input_<num>.png = a 100x100 image that can be pass to the MazeSolver as argument to load again this maze
* Press **g** to regenerate a new random maze     
* Press **c** to clear the window and have an empty board where you can put point manually
    * First click will place the start point
    * Second click will place the end point
    * All other click will place a wall
    * **Right click** on any placed point will delete it
* You can **Right click** to delete some wall and then press **SPACE** again to recalculate the solution
