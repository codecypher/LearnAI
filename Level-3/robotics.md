# Robotics

## Robotics - Modelling, Planning, and Control

### Introduction

_Robotics_ is concerned with the study of those machines that can replace human beings in the execution of a task, as regards both physical activity and decision making [1].

The goal of the introductory chapter is to point out the problems related to the use of robots in _industrial_ applications, as well as the perspectives offered by _advanced robotics_.

A classiﬁcation of the most common mechanical structures of robot manipulators and mobile robots is presented.

The topics of modelling, planning, and control are introduced which will be examined in the following chapters.

### 1.1 Robotics

Robotics is commonly defined as the science studying the intelligent connection between perception and action.

A _robotic system_ is in reality a complex system, functionally represented by multiple subsystems (Fig. 1.1).

Fig. 1.1. Components of a robotic system

The essential component of a robot is the _mechanical system_ endowed with a locomotion apparatus (wheels, crawlers, mechanical legs) and a manipulation apparatus (mechanical arms, end-eﬀectors, artiﬁcial hands). 

An example of the mechanical system in Fig. 1.1 consists of two mechanical arms (manipulation apparatus), each of which is carried by a mobile vehicle (locomotion apparatus). 

The realization of such a system refers to the context of design of articulated mechanical systems and choice of materials.

The capability to **exert an action** is provided by an _actuation system_ which animates the mechanical components of the robot.

The concept of an actuation system refers to the context of _motion control_ which concerns servomotors, drives, and transmissions.

The capability for **perception** is entrusted to a _sensory system_ which can acquire data on the internal status of the mechanical system (_proprioceptive sensors_ such as position transducers) as well as on the external status of the environment (_exteroceptive sensors_ such as force sensors and cameras).

The realization of the sensory system refers to the context of materials properties, signal conditioning, data processing, and information retrieval.

The capability for connecting action to perception in an intelligent fashion is provided by a _control system_ which can command the execution of the action in respect to the goals set by a task _planning_ technique, as well as of the constraints imposed by the robot and the environment.

The realization of such a system follows the same feedback principle devoted to _control_ of human body functions, possibly exploiting the description of the robotic system’s components called _modeling_.

Therefore, robotics is an interdisciplinary subject concerning the cultural areas of mechanics, control, computers, and electronics.


----------


## Robotics Technology

Robotics is an interdisciplinary sector of science and engineering dedicated to the design, construction and use of mechanical robots [2].

### What Is Robotics?

Robotics is the intersection of science, engineering, and technology that produces machines called robots that substitute for (or replicate) human actions [2].

While the overall world of robotics is expanding, a robot has some consistent characteristics:

1. Robots all consist of some sort of mechanical construction.

The mechanical aspect of a robot helps it complete tasks in the environment for which it is designed.

2. Robots need electrical components that control and power the machinery.

Essentially, an electric current (such as a battery) is needed to power a majority of robots.

3. Robots contain at least some level of computer programming.

Without a set of code telling it what to do, a robot would just be another piece of simple machinery.

Inserting a program into a robot gives it the ability to know when and how to carry out a task.

### What Is a Robot?

A _robot_ is a programmable machine that can complete a task [2].

The term _robotics_ describes the field of study focused on developing robots and automation.

These levels range from human-controlled bots that carry out tasks to fully-autonomous bots that perform tasks without any external influences.

In terms of etymology, the word ‘robot’ is derived from the Czech word robota which means “forced labor.”

The word first appeared in the 1920 play R.U.R. in reference to the play’s characters who were mass-produced workers incapable of creative thinking.

### Main Components of a Robot

Here are the main components of a robot [2]:

#### Control System

The _control system_ includes all of the components that make up a robot’s central processing unit.

Control systems are programmed to tell a robot how to utilize its specific components, similar in some ways to how the human brain sends signals throughout the body to complete a specific task.

#### Sensors

_Sensors_ provide a robot with stimuli in the form of electrical signals that are processed by the controller and allow the robot to interact with the outside world.

Common sensors found in robots include video cameras that function as eyes, photoresistors that react to light, and microphones that operate like ears.

#### Actuators

_Actuators_ are the components that are responsible for movement.

These components are made up of motors that receive signals from the control system and move in tandem to carry out the movement necessary to complete the assigned task.

Actuators can be made of a variety of materials and are commonly operated by use of compressed air (pneumatic actuators) or oil (hydraulic actuators), but come in a variety of formats to best fulfill their specialized roles.

#### Power Supply

Stationary robots may run on AC power through a wall outlet, but more commonly robots operate via an internal battery.

Safety, weight, replaceability and lifecycle are all important factors to consider when designing a robot’s power supply. 

### End Effectors

_End effectors_ are the physical (usually external) components that allow robots to complete their tasks.

Robots in factories often have interchangeable tools such as paint sprayers and drills, surgical robots may be equipped with scalpels, and other kinds of robots can be built with gripping claws or even hands for tasks such as deliveries, packing, bomb diffusion, etc.


### Types of Robots

Mechanical bots come in all shapes and sizes to efficiently carry out the task for which they are designed [3].

In general, there are five types of robots [3]:

#### Pre-Programmed Robots

Pre-programmed robots operate in a controlled environment where they do simple, monotonous tasks.

An example of a pre-programmed robot would be a mechanical arm on an automotive assembly line. The arm serves one function — to weld a door on, to insert a certain part into the engine, etc. — and it's job is to perform that task longer, faster, and more efficiently than a human.

#### Humanoid Robots

Humanoid robots are robots that look like and/or mimic human behavior.

These robots usually perform human-like activities (such as running, jumping and carrying objects), and are sometimes designed to look like us, even having human faces and expressions.

Two of the most prominent examples of humanoid robots are Hanson Robotics’ Sophia (in the video above) and Boston Dynamics’ Atlas.

#### Autonomous Robots

Autonomous robots operate independently of human operators.

These robots are usually designed to carry out tasks in open environments that do not require human supervision.

An example of an autonomous robot would be the Roomba vacuum cleaner which uses sensors to roam throughout a home freely.

#### Teleoperated Robots

Teleoperated robots are mechanical bots controlled by humans.

These robots usually work in extreme geographical conditions, weather, circumstances, etc.

Examples of teleoperated robots are the human-controlled submarines used to fix underwater pipe leaks during the BP oil spill or drones used to detect landmines on a battlefield.

#### Augmenting Robots

Augmenting robots either enhance current human capabilities or replace the capabilities a human may have lost.

Some examples of augmenting robots are robotic prosthetic limbs or exoskeletons used to lift hefty weights.


## Common Types of Robots for Manufacturing

There are four commmon types of industrial robots for manufacturing [4]:

### 1. Articulated Robots

- Pick and Place
- Machine Tending
- Assembly
- Welding
- Packaging
- Palletizing
- Inspection
- Material Removal
- Dispensing

An articulated robot is the type of robot that comes to mind when most people think about robots.

Much like CNC mills, articulated robots are classified by the number of points of rotation or axes they have.

The most common is the 6-axis articulated robot, but there are also 4- and 7-axis units on the market.

Flexibility, dexterity, and reach make articulated robots ideally suited for tasks that span non-parallel planes, such as machine tending.

Articulated robots can also easily reach into a machine tool compartment and under obstructions to gain access to a workpiece (or even around an obstruction, in the case of a 7-axis robot).

Sealed joints and protective sleeves allow articulated robots to excel in clean and dirty environments alike. The potential for mounting an articulated robot on any surface (such as a ceiling, a sliding rail) accommodates a wide range of working options.

The sophistication of an articulated robot comes with a higher cost compared to other robot types with similar payloads. And articulated robots are less suited than other types of robots for very high-speed applications due to their more complex kinematics and relatively higher component mass.

### 2. SCARA Robots

- Pick and Place
- Assembly
- Inspection
- Packaging
- Dispensing

A _Selective Compliance Articulated Robot Arm (SCARA)_ is a good (and cost-effective) choice for performing operations between two parallel planes (such as transferring parts from a tray to a conveyor).

SCARA robots excel at vertical assembly tasks such as inserting pins without binding due to their vertical rigidity.

SCARA robots are lightweight and have small footprints, making them ideal for applications in crowded spaces. They are also capable of very fast cycle times.

Due to their fixed swing arm design, which is an advantage in certain applications, SCARA robots face limitations when it comes to tasks that require working around or reaching inside objects such as fixtures, jigs, or machine tools within a work cell.

### 3. Delta Robots

- Pick and Place
- Assembly
- Inspection

Delta robots or _spider robots_ use three base-mounted motors to actuate control arms that position the wrist.

Basic delta robots are 3-axis units, but 4- and 6-axis models are also available.

By mounting the actuators on or close to the stationary base instead of at each joint (as in the case of an articulated robot), a delta robot’s arm can be very lightweight. This allows for rapid movement which makes delta robots ideal for very high-speed operations involving light loads.

An important thing to note as you compare delta robots to other robot types:

Reach for delta robots is typically defined by the diameter of the working range, as opposed to the radius from the base, as in the case of articulated and SCARA units.

For example, a delta robot with a 40” reach would only have half the reach (20” on a radius) of a 40” articulated or SCARA unit.

### 4. Cartesian Robots

- Pick and Place
- Dispensing
- Assembly
- Inspection

Cartesian robots typically consist of three or more linear actuators assembled to fit a particular application.

Positioned above a workspace, cartesian robots can be elevated to maximize floor space and accommodate a wide range of workpiece sizes.

When placed on an elevated structure suspended over two parallel rails, cartesian robots are referred to as _gantry robots_.

Cartesian robots typically use standard linear actuators and mounting brackets, minimizing the cost and complexity of any “custom” cartesian system.

Higher capacity units can also be integrated with other robots (such as articulated robots) as “end- effectors” to increase system capabilities.

The custom nature of cartesian robots can make design, specification, and programming challenging or out of reach for smaller manufacturers intent on a “DIY” approach to robotics implementation.

Cartesian robots are unable to reach into or around obstacles easily.
Their exposed sliding mechanisms make them less suited for dusty/dirty environments.


## References

[1]: B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo, Robotics - Modelling, Planning, and Control, ISBN: 978-1-84628-641-4, Springer, 2010.

[2]: [Robotics Technology](https://builtin.com/robotics)

[3]: [Definition, Types and Components of Robotics](https://benchpartner.com/definition-types-and-components-of-robotics-robot#google_vignette)

[4]: [4 Types of Robots Every Manufacturer Should Know](https://www.nist.gov/blogs/manufacturing-innovation-blog/4-types-robots-every-manufacturer-should-know)
