This page describes feedback from particles to grids and vice versa.
Please enable the compilation option [[FEEDBACK | Installation: Simulation-Options#FEEDBACK]].


## Compilation Options

Related options:
[[PARTICLE | Installation: Simulation-Options#PARTICLE]], &nbsp;
[[FEEDBACK | Installation: Simulation-Options#FEEDBACK]] &nbsp;


## Runtime Parameters
[[Runtime-Parameters:-Feedback]]

Other related parameters:

## Remarks

### Add User-defined Feedback
Follow the steps below to define your feedback when
[[adding a new simulation | Adding-New-Simulations]] named `NewProblem`.

1. Go to the new test problem folder and copy the feedback template.

    ```bash
    cd src/TestProblem/Hydro/NewProblem
    cp ../../../Feedback/User_Template/FB_User_Template.cpp FB_NewProblem.cpp
    ```

2. Edit the feedback source file `FB_NewProblem.cpp`.
    1. Rename `User_Template` as `NewProblem`.

    2. Follow the example `src/TestProblem/Hydro/Plummer/FB_Plummer.cpp` to edit
       `FB_Init_NewProblem()`, `FB_End_NewProblem()`, and `FB_NewProblem()`.

3. Edit the problem source file `Init_TestProb_Hydro_NewProblem.cpp` to enable this new feedback.

    1.  Put the following function prototype on the top of this file.

        ```C++
        #ifdef FEEDBACK
        void FB_Init_NewProblem();
        #endif
        ```

    2. Set the feedback function pointer in `Init_TestProb_Hydro_NewProblem()`.

    ```C++
    #  ifdef FEEDBACK
    FB_Init_User_Ptr = FB_Init_NewProblem;
    #  endif
    ```

4. Make sure to enable `FEEDBACK` in `Makefile` and `FB_USER` in `Input__Parameter`.


<br>

## Links
* [[Main page of Runtime Parameters | Runtime Parameters]]