## Rigid Body Motion Prediction with Planar Non-convex Contact Patch

Jiayin Xie 1 and Nilanjan Chakraborty 2

Abstract -We present a principled method for motion prediction via dynamic simulation for rigid bodies in intermittent contact with each other where the contact is assumed to be a planar non-convex contact patch. The planar non-convex contact patch can either be a topologically connected set or disconnected set. Such algorithms are useful in planning and control for robotic manipulation. Most work in rigid body dynamic simulation assume that the contact between objects is a point contact, which may not be valid in many applications. In this paper, by using the convex hull of the contact patch, we build on our recent work on simulating rigid bodies with convex contact patches, for simulating motion of objects with planar non-convex contact patches. We formulate a discrete-time mixed complementarity problem where we solve the contact detection and integration of the equations of motion simultaneously. Thus, our method is a geometrically-implicit method and we prove that in our formulation, there is no artificial penetration between the contacting rigid bodies. We solve for the equivalent contact point (ECP) and contact impulse of each contact patch simultaneously along with the state, i.e., configuration and velocity of the objects. We provide empirical evidence to show that our method can seamlessly capture transition between different contact modes like patch contact to multiple or single point contact during simulation.

## I. INTRODUCTION

Rigid body motion prediction via dynamic simulation is a key enabling technology in solving robotic manipulation with multi-fingered hands, vibratory plates, and parts feeder design [1], [2], [3], [4]. Many robotic manipulation tasks, e.g., extrinsic manipulation, manipulation by vibrating plates, involve point and surface contacts between the rigid body that is being manipulated and a flat plane on which the body rests [1], [3], [5]. Furthermore, the occurrence of multiple intermittent contacts makes the prediction of the motion more complicated. There are applications in which the contact between two objects may be over a patch that can be modeled as a non-convex set. For example, Figure 1 shows a robot manipulator manipulating a T-shaped bar where the contact between the ground and the bar is a planar non-convex set. Such situations may arise when a robot manipulator with a parallel jaw gripper is trying to reconfigure a heavy bar with support from the table, so that it does not have to support the full weight. State-of-the-art dynamic simulation algorithms that can be used to predict motions during planning, usually assume point contact between two objects (except [6], [7]), which is clearly violated in Figure 1. There are no wellprincipled approaches to predict the effect of applying a

1 Jiayin Xie is with the Department of Mechanical Engineering, Stony Brook University, 100 Nicolls Rd, Stony Brook, NY 11794 jiayin.xie@stonybrook.edu force/torque on the bar. In this paper, we seek to develop principled algorithms for predicting motion of rigid bodies in intermittent contact (via dynamic simulation), where the contacts can be modeled as a planar non-convex set.

2 Nilanjan Chakraborty is with the Department of Mechanical Engineering, Stony Brook University, 100 Nicolls Rd, Stony Brook, NY 11794 nilanjan.chakraborty@stonybrook.edu

Fig. 1: (Left) A T-shaped bar on planar surface is manipulated by a gripper while being supported on the plane, (Right) where the planar contact between T bar and support is a nonconvex T-shaped patch. The red line shows the convex hull for the contact patch.

<!-- image -->

Figure 2 shows the key types of contact between objects. Most existing mathematical models for motion of objects with intermittent contact like Differential Algebraic Equation (DAE) models [8] and Differential Complementarity Problem (DCP) models [9], [10], [11] assume the contact between the two objects is a single point contact (top left in Figure 2). However, for convex contact patch (middle row in Figure 2), the point contact assumption is not valid. In such case, multiple contacts point are usually chosen in an ad hoc manner, which can lead to inaccuracies in simulation (Please see [6] for example scenarios). Recently, we developed an approach [6] to simulate contacting rigid bodies with convex contact patches (line and surface contact). In [7], we develop an approach for simulating contacting bodies where the contact patch is non-convex but can be modeled as a union of convex sets (bottom row, right column in Figure 2). In this paper, we focus on simulating bodies with planar non-convex contact patch, where the non-convex contact patch may not be a union of convex sets. The contact can be multiple point contacts or a general planar non-convex patch contact (top row, right column and bottom row in Figure 2). Such situations arise when a robot is manipulating objects supported by a horizontal plane. For a single convex contact patch, we know that there exists a unique point on the contact surface where the integral of total moment due to normal force acting on this point is zero. This point is used to model line or surface contact as a point contact and thus it is called the equivalent contact point (ECP) [6]. Using the concept of ECP, in [6], we present a principled method for simulating intermittent contact with convex contact patches

Fig. 2: Different types of contact between one object with a flat surface. Our focus in this paper is on simulating rigid bodies with type of contact shown in last row and first row, pane (b).

<!-- image -->

(line and surface contact). This method solves for the ECP as well as the contact impulses by incorporating the collision detection within the dynamic simulation time step . This method is called the geometrically implicit time-stepping method because the geometric information of contact points and contact normal are solved as a part of the numerical integration procedure. In [7], for non-convex contact patches that can be modeled as a union of convex sets, we use an ECP to model the effect of each convex contact patch and solve for the ECP and its associated contact wrenches on each contact patch separately. However, the limitation of this method was that the force/moment distribution and the ECP was non-unique, although the state of the object was unique. Furthermore, if there are more than three convex sets forming the non-convex patch, the force/moment in some of the contact patches may become zero.

In this paper, we extend the method in [6], by using the convex hull of the contact patch for modeling the contact constraints in the equations of motion. Although, we have intermittent contact and the contact patch may change (even topologically, we can go from a connected non-convex patch to multiple point contact), we do not need to form the convex hull of the contact patch during the simulation depending on the contact mode. Instead, we use the convex hull of the non-convex object that is being manipulated. And since we solve the collision detection problem simultaneously with the equations of motion (i.e., our method is geometrically implicit), we can ensure that the convex hull of the contact patch will always be automatically obtained through our contact detection constraints. Note that distinct from [6], the ECP may not be a point within the physical contact region (but it will be a point within the convex hull of the contact region). We prove that even though we are modeling a non-convex contact patch with an equivalent contact point that may not lie within the patch, the contact constraints are always satisfied at the end of the time-step and there is no artificial penetration between the objects. We show simulation results validating our approach with our previous models [7], [12]. We also present simulation results showing that the object can seamlessly transition among different contact modes like non-convex patch contact, multiple point contact, line contact, and single point contact.

## II. RELATED WORK

In this section, we present the related work in rigid body dynamic simulation with a focus on methods for dealing with intermittent contact. There is also a substantial body of work on development of discretization schemes for integrating and simulating rigid body motion that we do not discuss here (please see the literature on variational integrators [13], [14], [15] and references therein). We model the continuous time dynamics of rigid bodies that are in intermittent contact with each other as a Differential Complementarity Problem (DCP). Let u ∈ R n 1 , v ∈ R n 2 and let g : R n 1 × R n 2 → R n 1 , f : R n 1 × R n 2 → R n 2 be two vector functions and the notation 0 ≤ x ⊥ y ≥ 0 imply that x is orthogonal to y and each component of the vectors is non-negative.

Definition 1: The differential (or dynamic) complementarity problem [16] is to find u and v satisfying

<!-- formula-not-decoded -->

Definition 2: The mixed complementarity problem is to find u and v satisfying

<!-- formula-not-decoded -->

If the functions f and g are linear, the problem is called a mixed linear complementarity problem (MLCP), otherwise, the problem is called a mixed nonlinear complementarity problem (MNCP). Our continuous time dynamics model is a DCP whereas our discrete-time dynamics model is a MNCP.

The DCP model formulates the intermittent contact between bodies in motion as a complementarity constraint [17], [18], [19], [20], [21], [22], [23]. DCP models are solved numerically with time-stepping schemes. The time-stepping problem is: given the state of the system and applied forces, compute an approximation of the system one time step into the future. Solving this problem repeatedly will give an approximate solution to the equations of motion.

There are different assumptions for forming the discrete equations of motion, which makes the system Mixed Linear Complementarity problem (MLCP) [24], [25] or mixed nonlinear complementarity problem (MNCP) [26], [27]. The MLCP problem linearizes the friction cone constraints and the distance function between two bodies (which is a nonlinear function of the configuration), sacrificing accuracy for speed. Depending on whether the distance function is approximated, the time-stepping schemes can also be divided into geometrically explicit schemes [18], [20] and geometrically implicit schemes [26].

In geometrically explicit schemes, at the current state, a collision detection routine is called to determine separation or penetration distances between the bodies, but this information is not incorporated as a function of the unknown future state at the end of the current time step. A goal of a typical time-stepping scheme is to guarantee consistency of the dynamic equations and all model constraints at the end of each time step. However, since the geometric information is obtained and approximated only at the start of the current time-step, then the solution will be in error. Apart from being geometrically explicit, most of the existing complementaritybased dynamic simulation methods and software also assume point contact between objects [28], [29], [30], [31], [32], [33]. A patch contact is usually approximated by ad hoc choice of 3 contact points on the contact patch. In [6], we compared our non-point contact model with two popular point-based models, namely, Open Dynamic Engine (ODE) [29] and Bullet [28] in a pure translation task with a square contact patch where the analytic closed-form solution is known. We showed that our results matched the theoretical results, and was more accurate compared to ODE and Bullet. Thus, in [6], [27], we used a geometrically implicit time stepping scheme for solving convex contact patches problem, which is also the method used in this paper. The resulting discrete time problem is a MNCP.

## III. DYNAMIC MODEL FOR RIGID BODY SYSTEMS

In complementarity methods, the dynamic simulation of intermittent unilateral contact between two rigid objects can be modeled by a geometrically implicit optimization-based time-stepping scheme. Note that the contact between objects is a planar contact patch, which can be either convex or non-convex. The dynamic model is made up of the following parts: (a) Newton-Euler equations (b) kinematic map relating the generalized velocities to the linear and angular velocities (c) friction law and (d) non-penetration constraints. The parts (a), (b) form a system of ordinary differential equations and they are standard for any complementarity-based formulation. Part (c) can be written as a system of complementarity constraints, which is based on Coulomb friction law using the maximum work dissipation principle. Part (d) incorporates the geometry of contact set as system of complementarity constraint [27], [6], [7].

To describe the dynamic model mathematically, we will introduce some notation first. Let q be the position of the center of mass of the object and the orientation of the object ( q can be 6 × 1 or 7 × 1 vector depending on the representation of the orientation). We will use unit quaternion to represent the orientation unless otherwise stated. The generalized velocity ν is the concatenated vector of linear ( v ) and spatial angular ( s ω ) velocities. The effect of the contact patch is modeled as point contact of equivalent contact points (ECPs) a 1 or a 2 on two objects. Let λ n be the magnitude of normal contact force, λ t and λ o be the orthogonal components of the friction force on the tangential plane, and λ r be the frictional moment about the contact normal.

## A. Newton-Euler equations of motion

<!-- formula-not-decoded -->

where M ( q ) = [ m I 3 0 0 s I cm ] is a symmetric, positive definite 6 × 6 matrix, which contains mass matrix m I 3 ( I 3 is a 3 × 3 identity matrix) and inertia matrix s I cm = R I cm R T . Here R is the 3 × 3 rotation matrix from body frame to world frame and I cm is the inertia matrix in the body frame. λ app is the 6 × 1 vector of external forces (including gravity) and moments, λ vp is the 6 × 1 vector of Coriolis and centripetal forces. W n , W t , W o and W r are dependent on configuration q and ECP ( a 1 or a 2 ), and map the normal contact forces, frictional forces and moments to the world reference frame:

<!-- formula-not-decoded -->

where ( n , t , o ) are unit vectors of contact frame and r is the vector from center of gravity to the ECP, in the world frame.

## B. Kinematic map

<!-- formula-not-decoded -->

where matrix G maps the generalized velocity ν to the time derivative of the position and orientation ˙ q .

## IV. MODELING PLANAR NON-CONVEX PATCH CONTACT

In this section, we will present our method for modeling a planar non-convex contact patch. Although, we will present the equations here in a more general manner, for concreteness, one can think that one object is a non-convex object and the other object is a plane (or a face of a polyhedron). This is the scenario where planar non-convex contact patch is easy to visualize and this situation is quite prevalent in robotics. Let F and G be the two objects, where, without loss of generality, the object F is the non-convex object. When two objects F and G have planar contact, the planar contact patch S is a non-empty finite subset of line or plane. We will use the convex hull of object F , denoted by Conv ( F ) to model the non-convex object F (this will be justified later in the section). We will now present the contact constraints for non-penetration of rigid bodies.

## A. Non-penetration constraints

In complementarity-based formulation of dynamics, the contact constraint for a potential contact is written as

<!-- formula-not-decoded -->

where ψ n ( q , t ) is the gap function for the contact with the property ψ n ( q , t ) &gt; 0 for separation, ψ n ( q , t ) = 0 for touching and ψ n ( q , t ) &lt; 0 for interpenetration. Note that there is usually no closed form expression for ψ n ( q , t ) . Thus, usually, a call is made to a collision detection module that provides information on the distance function and a first order approximation of the above equation is usually used in the discrete-time formulation of equations of motion, which can lead to inaccuracies in motion prediction [27].

In [27], we presented a method for incorporating the geometry of the contacting objects so that, we make sure that Equation (4) is satisfied exactly at the end of the time step and the contact points at the end of the time step are obtained. In [6], we showed that this method actually computes the ECP when the contact patch is a convex contact patch. We will now show that the contact constraints presented below allows us to compute the ECP of a non-convex contact patch as part of the integration of the equations of motion.

We assume that the convex hull of F , i.e., Conv ( F ) , and G are described by the intersecting convex inequalities f i ( x ) ≤ 0 , i = 1 , ..., m , and g j ( x ) ≤ 0 , j = m + 1 , ..., n respectively. Note that each individual convex constraint f i ( x ) = 0 describes the boundary of the convex hull. Note that multi-point contact, single point contact and convex patch contact are all special cases of the contact that we are considering. Let a 1 and a 2 be the pair of equivalent contact points for Conv ( F ) and G respectively. Note that, in general, a 1 may not be a point in F .

We rewrite the contact condition (Equation (4)) as a complementarity condition, and combine it with an optimization problem to find the closest points. Note that when objects are separate, the equivalent contact points a 1 and a 2 are solved as pair of closest points on the convex hull of F and G . However, this does not lead to any inaccuracies since separation of Conv ( F ) from the plane G implies separation of F and G and vice-versa. When objects have contact, a 1 and a 2 are solved as touching solution which prevents penetration between objects.

The convex inequality has the property that for any point x , the point lies inside the object when f ( x ) &lt; 0 , on the boundary of object when f ( x ) = 0 , and outside the object when f ( x ) &gt; 0 . Thus, the contact condition (Equation (4)) can be rewritten as either one of the following two complementarity constraints [27]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where a 1 and a 2 are given by a solution to the following minimization problem:

<!-- formula-not-decoded -->

As shown in [27], based on a modification of the KKT conditions, we can show that the ECPs need to satisfy the algebraic and complementarity constraints given below to solve the optimization problem above (Equation (7)).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

where ∇C ( F i , a 1 ) = ∇ f k ( a 1 ) + ∑ m i = k l i ∇ f i ( a 1 ) , k represents the index of any one of the active constraints (i.e., the surface on which the ECP a 1 , a 2 lies). We will also need an additional complementarity constraint (either Equation (5) or Equation (6)) to prevent penetration:

<!-- formula-not-decoded -->

Equations (8) ∼ (12) together gives the constraints that the equivalent contact points a 1 and a 2 for should satisfy for ensuring no penetration between the objects. We prove this formally in Proposition 2 . We first prove that the use of the convex hull ensures that the ECP that we compute is within the convex hull of the contact patch.

Proposition 1: By using the convex hull of the object F to formulate the contact constraints, we ensure that we compute the ECP within the convex hull of the contact patch.

Proof: Due to lack of space, we present a sketch of the proof idea here. The convex hull contains the set of all the extreme points 1 of the object. For a non-convex object contacting with a plane, the set of extreme points are the only points that can potentially contact the plane. Therefore, using the convex hull description ensures that we are capturing the set of all points that can be in contact. So when we are solving for the ECP, it will be in the convex region defined by the active constraints which is essentially the convex hull of the set of contacting points.

Proposition 2: When using Equations (8) to (12) to model the contact between convex hulls for two objects, we get the solution for ECPs as the closest points on the boundary of convex hulls respectively when objects are separate. When objects have planar contact, we will get touching solution which prevents penetration.

Proof: Because of lack of space, we do not provide the full proof here. The proof essentially follows from the arguments of the proof shown in [27] and [6], with minor modification to consider the convex hull of F instead of F .

## B. Friction Model

Our friction model is based on the maximum power dissipation principle and generalized Coulomb's friction law.

The effect of the patch can be modeled as point contact based on the ECP a 1 or a 2 :

<!-- formula-not-decoded -->

where v t and v o are the tangential components of the relative velocity at ECP of the contact patch, v r is the relative angular velocity about the normal at ECP. e t , e o and e r is the given positive constants defining the friction ellipsoid and µ represents the coefficient of friction at the contact [34], [10]. This constraint is the elliptic dry friction condition suggested in [34] based upon evidence from a series of contact experiments. This model states that among all the possible contact forces and moments that lie within the friction ellipsoid, the forces and moment that maximize the power dissipation at the contact (due to friction) are selected.

1 The extreme point of a set is a point satisfying the following property: There exists a hyperplane passing through the point such that all points in the set lies on one side of the hyperplane.

This argmax formulation of the friction law has a useful alternative formulation [35]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where W T ( . ) are dependent on ECP for the contact patch and σ is the magnitude of the slip velocity on the contact patch.

## C. Time-stepping Formulation

We use a velocity-level formulation and an Euler timestepping scheme to discretize the above system of equations. Let t u denote the current time and h be the duration of the time step, the superscript u represents the beginning of the current time and the superscript u + 1 represents the end of the current time. Using ˙ ν ≈ ( ν u +1 -ν u ) /h , ˙ q ≈ ( q u +1 -q u ) /h and writing forces as impulses, we get the discretized Newton-Euler equations and kinematic map:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where impulse p ( . ) = hλ ( . ) , the contact impulse P u +1 c is:

<!-- formula-not-decoded -->

where W t , W n , W o , W r are dependent on ECPs at the end of time step u +1 .

We discretize contact constraints (Equations (8) ∼ (12)) and friction model (Equations (14) and (15)) by writing forces λ ( . ) into impulses p ( . ) . Furthermore, the unknown contact impulses in Equations (14), (15) and unknown ECPs in Equations (8) ∼ (12) are at the end of time step u +1 .

## D. Summary of geometrically implicit time-stepping scheme

As stated earlier, our dynamic model is composed of (a) Newton-Euler equations (Equation (16)), (b) kinematic map between the rigid body generalized velocity and the rate of change of the parameters for representing position and orientation (Equation (17)), (c) contact model which gives the constraints that the equivalent contact points a 1 and a 2 should satisfy for ensuring no penetration between the objects (Equations (8) ∼ (12)). (d) friction model which gives the constraints that contact wrenches should satisfy (Equations (14) and (15)). Thus, we have a coupled system of algebraic and complementarity equations (mixed nonlinear complementarity problem) that we have to solve.

## V. SIMULATION RESULTS

In this section, we evaluate the performance of our proposed method on two example problems. All the simulations are run in MATLAB on a MacBook Pro with 2.6 GHZ processor and 16 GB RAM.

## A. Comparison with existing methods

We first consider the problem of predicting motion of a square desk with four legs pushed by two grippers, where the contact patch between desk feet and support is a union of four small squares (see Figure 3a). Such problems are useful for robots navigating or rearranging furniture in domestic environments. The dimension for the square desk is length L = 0 . 5 m , the length for each small square is L s = 0 . 06 m and height of desk's CM is H = 0 . 45 m . The mass of desk is m = 15 kg and the gravity's acceleration is g = 9 . 8 m/s 2 .

We compare the convex hull-based contact detection method presented in this paper with method in [7] and [12]. Note that we apply forces, so as to ensure sliding without toppling. Therefore, the dynamic model for sliding motion that we proposed in [12] can be applied here. In [12], we have shown that for sliding-only motion, the discrete-time equations of motion can be reduced to a system of four quadratic equations. Since the contact is a union of four disjoint squares, we can also use the method in [7], where, we consider each non-penetration constraint between each contact patch and the ground separately.

The time step chosen for all the simulations is h = 0 . 01 s and simulation time is 4 s . The coefficient of friction between desk and support is µ = 0 . 22 and the given constants for friction ellipsoid are e t = e o = 1 , e r = 0 . 1 m . As shown in Figure 3a, the desk slides on the support. The initial position of CM is q x = q y = 0 m , q z = 0 . 45 m and orientation about normal axis is θ z = 0 . The initial velocity is v x = 0 . 3 m/s , v y = 0 . 2 m/s , w z = 0 . 5 rad/s . The external forces and moments from grippers exerted on the desk is periodic, f x = 22 . 5 sin(2 πt ) + 22 . 5 N , f y = 22 . 5 cos(2 πt ) + 22 . 5 N , τ z = 2 . 1 cos(2 πt ) Nm , where t ∈ [0 , 4] s .

In Figure 3b, we plot the snapshot for the contact patch during the motion. It can be seen from the figure, that the table translates as well as rotates during motion. The ECP is marked by a red cross and it can be seen that the ECP is not within the contact patch and it is also not below the center of mass of the table (which matches the intuition, since the table is rotating). In Figure 3c, we plot the velocity of the desk ( v x and w z ) during the motion. In addition, we plot the difference between solutions of quadratic model and convex hull method, and difference between solutions of MNCP model and convex hull method. As Figure 3c illustrates, the differences for v x and w z are within 1e-8, which validates the accuracy of convex hull method.

Furthermore, the average time the model in [12] spends for each time step is 0 . 0022 s , the time our proposed method method spends is 0 . 0053 s (which is 2 . 4 times than [12]), and the time the model in [7] spends is 0 . 0487 s (which is more than 22 times than quadratic model's and 9 times than convex hull method). To summarize, proposed method simplify the model in [7] greatly by modeling multiple contact patches with a single patch and therefore is much more efficient. The model in [12], although faster is valid only for sliding and cannot be applied to situations where the object may topple.

<!-- image -->

(a) A four-legged desk on ground is pushed by two grippers (Left), where the contact between desk feet and ground is a union of four squares (Right). We get the convex hull for the contact patch (red square).

(b) The snapshot for the contact patch between desk feet and ground during the motion. The ECP is shown in red dot.

<!-- image -->

(c) (Top Left) The solution of translational velocity from [12] ( v x Q ), from [7] ( v x N ) and proposed method ( v x C ). (Bottom Left) The difference between v x C and v x N , and difference between v x C and v x Q . (Top Right) The angular velocity w z Q , w z N and w z C , (Bottom Right) and the differences between them.

<!-- image -->

Fig. 3: Comparison of the proposed method with [7], [12].

<!-- image -->

(a) (Left) We plot the applied torques ( τ x , τ y , τ z ) from the gripper exerting on the T-shaped bar along with the time. (Right) In addition, we plot the applied force ( F x , F y ) acting on the bar.

(b) (Left) The two-point contact which is non-convex between the bar and ground is replaced by the convex line contact in red. (Right) The coordinates of ECP ( a x , a y , a z ) during the motion.

<!-- image -->

Fig. 4: Simulation for the motion of T-shaped bar example based on the proposed method.

<!-- image -->

## B. Simulations of the T-shaped bar

This example is used to illustrate that our method allows objects to automatically transition between different contact modes (surface, point, line and also making and breaking of contact), while ensuring the objects do not penetrate. As Figure 1 illustrates, the planar contact patch between Tshaped bar and the support is non-convex.

The dimensions of the bar are given in Figure 1. The mass of the bar is 2 kg , the other parameters like gravity and friction parameters are the same as in the first example. The time step chosen is h = 0 . 01 s and the total simulation time is t = 5 s . Figure 4a shows the applied forces and moments from the gripper acting on the bar, and Figure 4b demonstrates the coordinates of ECP (i.e., a x , a y , a z ). Note that the coordinate of ECP along z axis a z stays zero within the numerical tolerance of 1 e -12 during the motion. Thus, there is no penetration between the bar and ground. The snapshots show the transition of the bar from surface contact 4c to two-point contact 4d to another two-point contact with different pair of contact points 4e to a surface contact 4f and then rotation while having the surface contact.

All these transitions were automatically detected by our algorithm.

## VI. CONCLUSION

In this paper we present a geometrically implicit timestepping method for solving dynamic simulation problems with planar non-convex contact patches. In our model, we use a convex hull of the non-convex object and combine the collision detection with numerical integration of equations of motion. This allows us to solve for an equivalent contact point (ECP) in the convex hull of the non-convex contact patch as well as the contact wrenches simultaneously. We prove that although we model the contact patch with an ECP, the non-penetration constraints at the end of the time-step are always satisfied. We present numerical simulation motion prediction for two example problems that are representative of applications in robotic manipulation. The results demonstrate that our method can automatically transition among different contact modes (non-convex contact patch, point, and line).

## REFERENCES

- [1] D. Reznik and J. Canny, 'A flat rigid plate is a universal planar manipulator,' in Proceedings of IEEE International Conference on Robotics and Automation , vol. 2, May 1998, pp. 1471-1477.
- [2] P. Song, J. Trinkle, V . Kumar, and J. Pang, 'Design of part feeding and assembly processes with dynamics,' in IEEE Intl. Conf. on Robotics and Automation , New Orleans, LA, May 2004, pp. 39 - 44.
- [3] T. H. Vose, P. Umbanhowar, and K. M. Lynch, 'Friction-induced lines of attraction and repulsion for parts sliding on an oscillated plate,' IEEE Transactions on Automation Science and Engineering , vol. 6, no. 4, pp. 685-699, Oct 2009.
- [4] S. Berard, B. Nguyen, K. Anderson, and J. Trinkle, 'Sources of error in a simulation of rigid parts on a vibrating rigid plate,' ASME Journal of Computational and Nonlinear Dynamics , vol. 5, no. 4, p. 041003, 2010.
- [5] N. C. Dafle, A. Rodriguez, R. Paolini, B. Tang, S. S. Srinivasa, M. A. Erdmann, M. T. Mason, I. Lundberg, H. Staab, and T. A. Fuhlbrigge, 'Extrinsic dexterity: In-hand manipulation with external forces,' in Proceedings of IEEE International Conference on Robotics and Automation (ICRA) , 2014, pp. 1578-1585.
- [6] J. Xie and N. Chakraborty, 'Rigid body dynamic simulation with line and surface contact,' in 2016 IEEE International Conference on Simulation, Modeling, and Programming for Autonomous Robots (SIMPAR) , Dec 2016, pp. 9-15.
- [7] --, 'Rigid body dynamic simulation with multiple convex contact patches,' in proc. of ASME IDETC &amp; International Conference on Multibody Systems, Nonlinear Dynamics, and Control (IDETC/MSNDC 2018) . American Society of Mechanical Engineers, Aug 2018, pp. V006T09A002-V006T09A002.
- [8] E. J. Haug, S. C. Wu, and S. M. Yang, 'Dynamics of mechanical systems with coulomb friction, stiction, impact and constraint additiondeletion theory,' Mechanism and Machine Theory , vol. 21, no. 5, pp. 401-406, 1986.
- [9] R. W. Cottle, J.-S. Pang, and R. E. Stone, The linear complementarity problem . SIAM, 2009, vol. 60.
- [10] J. C. Trinkle, J.-S. Pang, S. Sudarsky, and G. Lo, 'On dynamic multirigid-body contact problems with coulomb friction,' ZAMM-Journal of Applied Mathematics and Mechanics/Zeitschrift f¨ ur Angewandte Mathematik und Mechanik , vol. 77, no. 4, pp. 267-279, 1997.
- [11] F. Pfeiffer and C. Glocker, Multibody Dynamics with Unilateral Contacts . Wiley Inc., 2008.
- [12] J. Xie and N. Chakraborty, 'Dynamic models of planar sliding,' in Algorithmic Foundations of Robotics (WAFR), The 13th International Workshop on the . IFRR, 2018. [Online]. Available: https://drive. google.com/open?id=1OTNMP8HukTzQsBnAPtAC0obA8DSpQSxf
- [13] J. E. Marsden and M. West, 'Discrete mechanics and variational integrators,' Acta Numerica , vol. 10, pp. 357-514, 2001.
- [14] E. R. Johnson and T. D. Murphey, 'Scalable variational integrators for constrained mechanical systems in generalized coordinates,' IEEE Transactions on Robotics , vol. 25, no. 6, p. 1249, 2009.
- [15] M. Kobilarov, K. Crane, and M. Desbrun, 'Lie group integrators for animation and control of vehicles,' ACM Trans. Graph. , vol. 28, no. 2, pp. 16:1-16:14, May 2009. [Online]. Available: http://doi.acm.org/10.1145/1516522.1516527
- [16] F. Facchinei and J.-S. Pang, Finite-dimensional variational inequalities and complementarity problems . Springer Science &amp; Business Media, 2007.
- [17] P. Lotstedt, 'Mechanical systems of rigid bodies subject to unilateral constraints,' SIAM Journal on Applied Mathematics , vol. 42, no. 2, pp. 281-296, 1982.
- [18] M. Anitescu, J. F. Cremer, and F. A. Potra, 'Formulating 3d contact dynamics problems,' Mechanics of Structures and Machines , vol. 24, no. 4, pp. 405-437, 1996.
- [19] J.-S. Pang and J. C. Trinkle, 'Complementarity formulations and existence of solutions of dynamic multi-rigid-body contact problems with coulomb friction,' Mathematical Programming , vol. 73, no. 2, pp. 199-226, 1996.
- [20] D. E. Stewart and J. C. Trinkle, 'An implicit time-stepping scheme for rigid body dynamics with inelastic collisions and Coulomb friction,' International Journal of Numerical Methods in Engineering , vol. 39, pp. 2673-2691, 1996.
- [21] T. Liu and M. Y. Wang, 'Computation of three-dimensional rigidbody dynamics with multiple unilateral contacts using time-stepping and Gauss-seidel methods,' IEEE Transactions on Automation Science and Engineering , vol. 2, no. 1, pp. 19-31, Jan. 2005.
- [22] E. Drumwright and D. A. Shell, 'Extensive analysis of linear complementarity problem (lcp) solver performance on randomly generated rigid body contact problems,' in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems , Oct 2012, pp. 5034-5039.
- [23] E. Todorov, 'Convex and analytically-invertible dynamics with contacts and constraints: Theory and implementation in mujoco,' in 2014 IEEE International Conference on Robotics and Automation (ICRA) , May 2014, pp. 6054-6061.
- [24] M. Anitescu and F. A. Potra, 'Formulating dynamic multi-rigid-body contact problems with friction as solvable linear complementarity problems,' Nonlinear Dynamics , vol. 14, no. 3, pp. 231-247, 1997.
- [25] --, 'A time-stepping method for stiff multibody dynamics with contact and friction,' International Journal for Numerical Methods in Engineering , vol. 55, no. 7, pp. 753-784, 2002.
- [26] J. E. Tzitzouris, 'Numerical resolution of frictional multi-rigid-body systems via fully implicit time-stepping and nonlinear complementarity,' Ph.D. dissertation, Johns Hopkins University, 2001.
- [27] N. Chakraborty, S. Berard, S. Akella, and J. Trinkle, 'A geometrically implicit time-stepping method for multibody systems with intermittent contact,' The International Journal of Robotics Research , vol. 33, no. 3, pp. 426-445, 2014.
- [28] E. Coumans, 'Bullet physics engine for rigid body dynamics,' http: //bulletphysics.org/.
- [29] R. Smith, 'Open dynamics engine ode. multibody dynamics simulation software,' http://www.ode.org/.
- [30] E. Todorov, T. Erez, and Y. Tassa, 'Mujoco: A physics engine for model-based control,' in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems , Oct 2012, pp. 5026-5033.
- [31] A. Tasora, R. Serban, H. Mazhar, A. Pazouki, D. Melanz, J. Fleischmann, M. Taylor, H. Sugiyama, and D. Negrut, 'Chrono: An open source multi-physics dynamics engine,' in International Conference on High Performance Computing in Science and Engineering . Springer, 2015, pp. 19-49.
- [32] J. Lee, M. X. Grey, S. Ha, T. Kunz, S. Jain, Y. Ye, S. S. Srinivasa, M. Stilman, and C. K. Liu, 'Dart: Dynamic animation and robotics toolkit,' The Journal of Open Source Software , vol. 3, no. 22, p. 500, 2018.
- [33] S. Berard, J. Trinkle, B. Nguyen, B. Roghani, J. Fink, and V. Kumar, 'davinci code: A multi-model simulation and analysis tool for multibody systems,' in Proceedings 2007 IEEE International Conference on Robotics and Automation . IEEE, 2007, pp. 2588-2593.
- [34] R. D. Howe and M. R. Cutkosky, 'Practical force-motion models for sliding manipulation,' The International Journal of Robotics Research , vol. 15, no. 6, pp. 557-572, 1996.
- [35] J. C. Trinkle, J. Tzitzouris, and J.-S. Pang, 'Dynamic multi-rigid-body systems with concurrent distributed contacts,' Philosophical Transactions of the Royal Society of London A: Mathematical, Physical and Engineering Sciences , vol. 359, no. 1789, pp. 2575-2593, 2001.