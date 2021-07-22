### Config-file structure
Each config-file contains a python dictionary of dictionaries converted into JSON-format. The top-level dictionary keys are ```"animation"```, ```"env"```, ```"interaction"```, ```"model"``` and ```"tasks"```. From these ```"animation"```, ```"env"``` and ```"model"``` values are used for initializing the corresponding objects. The ```"tasks"``` value is a list of movement tasks that the agents will execute during the animation. Finally, the ```"interaction"``` value is a string which tells what kind of interaction model will be used.

The dictionary structure:
```
{
	"animation": <init params>,
	"env": <init params>,
	"interaction", <string>,
	"model": <init params>,
	"tasks": <list of tasks>
}
```

#### Init params
The animation init params has several parameters for the animation figure plot adjustments. The env init params has parameters for the strength and width of the vector field that is responsible for the disturbance and for its visualization. For details on these I suggest to look at the config-files and the docstrings of the methods of ```Animation``` and ```Env``` classes. Here I will only go trough the model init params since the main focus of this project is on the algorithmic side.

The model init params structure:
```
{
        "accepted_error": <float>,
        "bond_strength": <float>,
        "max_speed": <float>,
        "target_distance": <float>,
        "time_delta": <float>
}
```

Parameters:  ```"accepted_error"``` is used for deciding if a task is done or not,  ```"bond_strength"``` is used for the strength of the pull towards the triangular shape of the formation, ```"max_speed"``` is the cut-off speed for all agents, ```"target_distance"``` is the target length of each side in the formation and finally ```"time_delta"``` is length parameter for single time unit.

#### Interactions and tasks
There are two kinds of interactions: ```"central_control"``` and ```"one_lead"```.

The tasks currently supported by the central control model are ```"reshape"```, ```"shift"``` and ```"turn"```. The one lead model supports: ```"reshape"```, ```"shift"```, ```"start_acceleration"``` and ```"apply_acceleration"``` tasks.

In the list of tasks each task has the following structure:
```
{
        "args": <list of argumeants>,
        "type": <string of movent type>
}
```
The arguments are fed to a method that corresponds to the given movement type. The task types and their arguments are listed below:

- ```"reshape"``` This task is meant to done first. It moves the agents into triangular formation from their initial random positions. For the centrally controlled model all agents move and for the model with one lead agent only the follower agents move.

	Args: ```speed (float)```

- ```"shift"``` This task moves the formation to given target point. For the centrally controlled model it will be finished when the center point of the formation is close enough to the target and for the one-lead model when the lead agent is. Under the hood this is actually done by keeping track of the undisturbed movement and doing course corrections. (So if the disturbance is big and the distance to the target is short the task might not finish properly.)

	Args: ```target_point (list of two floats)```, ```speed (float)```

- ```"turn"``` This task will turn the formation by given angle around its center point. It is only used by the centrally controlled model.

	Args: ```target_point (list of two floats)```, ```speed (float)```

- ```"start_acceleration"``` This task is meant to be used for initiating acceleration to given direction when the formation is at rest (mainly after the reshape task). It is only available for the one-lead model.

	Args: ```acceleration_type ("start")```, ```parameters (list)```

	where parameters are: ```strength (float)```, ```direction (list of two floats)```, ```duration (int)```, ```follow_speed (float)```

	For this both this task and the apply-acceleration that is explained below, ```duration``` is the number of time steps for which the acceleration is used and ```follow_speed``` is the speed parameter for the follower agents formation keeping movements.

- ```"apply_acceleration"``` This task is meant to be used when formation is already moving. It adds given amounts of tangential and normal acceleration to the velocity. It is only available for the one-lead model.

	Args: ```acceleration_type ("apply")```, ```parameters (list)```

	where parameters are: ```tangential (float)```, ```normal (float)```, ```duration (int)```, ```follow_speed (float)```

	Here ```tangential``` is the strength of the acceleration towards current velocity direction and ```normal``` towards the direction which is 90 degrees clockwise from it.
