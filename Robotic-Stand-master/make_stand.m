clear all
close all
clf
handle_axes= axes('XLim', [-1.8,1.8], 'YLim', [-1.8,1.8], 'ZLim', [0,1.4]); % sets the x,y and z axis

xlabel('x'); 
ylabel('y');
zlabel('z');

view(-130, 26);
grid on;
axis equal
camlight %changes the light of the diagram
axis_length= 0.05;

%% Root frame E
trf_E_axes= hgtransform('Parent', handle_axes); 
% The root-link transform should be created as a child of the axes from the
% beginning to avoid the error "Cannot set property to a deleted object".
% E is synonymous with the axes, so there is no need for plot_axes(trf_E_axes, 'E');

%% Link-0: Stand/base

trf_link0_E= make_transform([0, 0, 0], 0, 0, pi/2, trf_E_axes);
plot_axes(trf_link0_E, 'L_0', false, axis_length); 

trf_viz_link0= make_transform([0, 0, 0], 0, 0, 0, trf_link0_E); % Do not specify parent yet: It will be done in the joint
h(1)= link_box([0.05, 0.05, 0.01], trf_viz_link0, [0.823529 0.311765 0.117647]); 
plot_axes(trf_viz_link0, 'ls', true, axis_length); % V_{1-2}M


%% Link-1: Vertical Cylinder
trf_viz_link1= make_transform([0, 0, 0.2], 0, 0, 0); % 0.02->0.0525, Do not specify parent yet: It will be done in the joint
length1= 0.4;%length of upper block link 1
radius1= 0.005;%radius of upper block link 1

h(2)= link_cylinder(radius1, length1, trf_viz_link1, [0.8, 0, 0]); 
plot_axes(trf_viz_link1, 'lp', true, axis_length); % V_0


%% 
%%%% Link-2: Supporting Horizontal Cylinder 1
 trf_viz_link2= make_transform([0, 0, 0.3], 0, pi/2, 0); % 0.02->0.0525, Do not specify parent yet: It will be done in the joint
 length1= 0.1;%length of upper block link 1
 radius1= 0.005;%radius of upper block link 1
 
 h(3)= link_cylinder(radius1, length1, trf_viz_link2, [0.8, 0, 0]); 
  plot_axes(trf_viz_link2, 'lv', true, axis_length); % V_0

  %% Link-3: Vertical Cylinder_2
trf_viz_link3= make_transform([0, 0, -0.1], 0, 0, 0); % 0.02->0.0525, Do not specify parent yet: It will be done in the joint
length1= 0.2%length of upper block link 1
radius1= 0.005;%radius of upper block link 1

h(4)= link_cylinder(radius1, length1, trf_viz_link3, [0.8, 0, 0]); 
plot_axes(trf_viz_link3, 'lm', true, axis_length); % V_0 

 %% Link-4: Vertical Cylinder_3
trf_viz_link4= make_transform([0, 0, -0.1], 0, 0, 0); % 0, -0.05,0.2 Do not specify parent yet: It will be done in the joint
length1= 0.2%length of upper block link 1
radius1= 0.005;%radius of upper block link 1

h(5)= link_cylinder(radius1, length1, trf_viz_link4, [0.8, 0, 0]); 
plot_axes(trf_viz_link4, 'lfew', true, axis_length); % V_0 

%% 

% %% Link-5: Sphere
% trf_viz_link5 = make_transform([0, 0, 0], 0, 0,0);
% radius = 0.01;
% 
% h(5)= link_sphere(radius, trf_viz_link4, [0.8, 0, 0]); 
% plot_axes(trf_viz_link5, 'le', true, axis_length); % V_0 

% %% Link-6: Sphere
% trf_viz_link6 = make_transform([0, 0, 0], 0, 0,0);
% radius = 0.01;
% 
% h(5)= link_sphere(radius, trf_viz_link4, [0.8, 0, 0]); 
% plot_axes(trf_viz_link6, 'lf', true, axis_length); % V_0 


%% %% Joint 1: Links 0,1: Revolute
 j1_rot_axis_j1= [0,0,1]';
 j1_rot_angle= 0; %[-pi/2, pi/2];
 
 trf_joint1_link0= make_transform([0, 0, 0], 0, 0, 0, trf_link0_E); 
 trf_link1_joint1= make_transform_revolute(j1_rot_axis_j1, j1_rot_angle, trf_joint1_link0); 
 plot_axes(trf_link1_joint1, 'L_1', false, axis_length); 
 make_child(trf_link1_joint1, trf_viz_link1);
 
 %% 
 %  % Joint 2: Links 1,2: Fized
trf_link2_link1= make_transform([0, 0, 0], 0, 0, 0, trf_link1_joint1); 
 make_child(trf_link2_link1, trf_viz_link2);
 plot_axes(trf_link1_joint1, '', false, axis_length);
 
  %% %% %% Joint 2: Links 2,3: Revolute
 j2_rot_axis_j2= [0,1,0]';
 j2_rot_angle= 0; %[-pi/2, pi/2];
 
 trf_joint2_link2= make_transform([0.05, 0, 0.3], 0, -pi/8, 0, trf_link2_link1); 
 trf_link3_joint2= make_transform_revolute(j2_rot_axis_j2, j2_rot_angle, trf_joint2_link2); 
 plot_axes(trf_link3_joint2, 'L_wot', false, axis_length); 
 make_child(trf_link3_joint2, trf_viz_link3);
 
 j4_rot_axis_j4= [1,0,0]';
 j4_rot_angle= 0; %[-pi/2, pi/2];
 
 trf_joint2_link2= make_transform([0.05, 0, 0.3], 0, -pi/8, 0, trf_link2_link1); 
 trf_link3_joint2= make_transform_revolute(j2_rot_axis_j2, j4_rot_angle, trf_joint2_link2); 
 plot_axes(trf_link3_joint2, 'L_wot', false, axis_length); 
 make_child(trf_link3_joint2, trf_viz_link3);
 
 %trf_joint2_link2= make_transform([0.05, 0, 0.3], 0, -pi/8, 0, trf_link2_link1); 
 %trf_link3_joint2= make_transform_revolute(j2_rot_axis_j2, j2_rot_angle, trf_joint2_link2); 
 %plot_axes(trf_link3_joint2, 'L_wot', false, axis_length); 
 %make_child(trf_link3_joint2, trf_viz_link3)
 
 
  % %% %% Joint 3: Links 3,4: Revolute
 j3_rot_axis_j3= [0,1,0]';
 j3_rot_angle= 0; %[-pi/2, pi/2];
 
 trf_joint3_link3= make_transform([-0.05, 0, 0.3], 0, pi/8, pi/2, trf_link2_link1); %-pi/8, pi/2 causing discrepancy
 trf_link4_joint3= make_transform_revolute(j3_rot_axis_j3, j3_rot_angle, trf_joint3_link3); 
 plot_axes(trf_link4_joint3, 'L_not', false, axis_length); 
 make_child(trf_link4_joint3, trf_viz_link4);
 
 
 j5_rot_axis_j5= [1,0,0]';
 j5_rot_angle= 0; %[-pi/2, pi/2];
 
 trf_joint3_link3= make_transform([-0.05, 0, 0.3], 0, pi/8, 0, trf_link2_link1); %-pi/8
 trf_link4_joint3= make_transform_revolute(j5_rot_axis_j5, j5_rot_angle, trf_joint3_link3); 
 plot_axes(trf_link4_joint3, 'L_not', false, axis_length); 
 make_child(trf_link4_joint3, trf_viz_link4);
 
  % 

 %Animation: One joint at a time
for q1=[linspace(0, -pi/2, 30), linspace(-pi/2, pi/2, 30), linspace(pi/2, 0, 30)]
    set(handle_axes, 'XLim', [-0.2,0.2], 'YLim', [-0.2,0.2], 'ZLim', [0,0.4]);
    trf_q1= makehgtform('axisrotate', j1_rot_axis_j1, q1);
    set(trf_link1_joint1, 'Matrix', trf_q1);
    drawnow;
    pause(0.02);
end

for q2=[linspace(0, -pi/2, 30), linspace(-pi/2, pi/2, 30), linspace(pi/2, 0, 30)]
    set(handle_axes, 'XLim', [-0.2,0.2], 'YLim', [-0.2,0.2], 'ZLim', [0,0.4]);
    trf_q2= makehgtform('axisrotate', j2_rot_axis_j2, q2);
    set(trf_link3_joint2, 'Matrix', trf_q2);
    drawnow;
    pause(0.02);
end

for q3=[linspace(0, -pi/2, 30), linspace(-pi/2, pi/2, 30), linspace(pi/2, 0, 30)]
    set(handle_axes, 'XLim', [-0.2,0.2], 'YLim', [-0.2,0.2], 'ZLim', [0,0.4]);
    trf_q3= makehgtform('axisrotate', j3_rot_axis_j3, q3);
    set(trf_link4_joint3, 'Matrix', trf_q3);
    drawnow;
    pause(0.02);
end
for q4=[linspace(0, -pi/2, 30), linspace(-pi/2, pi/2, 30), linspace(pi/2, 0, 30)]
    set(handle_axes, 'XLim', [-0.2,0.2], 'YLim', [-0.2,0.2], 'ZLim', [0,0.4]);
    trf_q4= makehgtform('axisrotate', j4_rot_axis_j4, q4);
    set(trf_link3_joint2, 'Matrix', trf_q4);
    drawnow;
    pause(0.02);
end

for q5=[linspace(0, -pi/2, 30), linspace(-pi/2, pi/2, 30), linspace(pi/2, 0, 30)]
    set(handle_axes, 'XLim', [-0.2,0.2], 'YLim', [-0.2,0.2], 'ZLim', [0,0.4]);
    trf_q5= makehgtform('axisrotate', j5_rot_axis_j5, q5);
    set(trf_link4_joint3, 'Matrix', trf_q5);
    drawnow;
    pause(0.02);
end

% for a4=[linspace(0, -0.04, 30), linspace(-0.04, 0.04, 30), linspace(0.04, 0, 30)]
%     set(handle_axes, 'XLim', [-0.4,0.4], 'YLim', [-0.2,0.4], 'ZLim', [0,0.4]);
%     trf_a4= makehgtform('translate', j4_translation_axis_j4*a4);
%     set(trf_link4_joint3, 'Matrix', trf_a4);
%     drawnow;
%     pause(0.02);
% end
% %for q5=[linspace(0, -pi/2, 30), linspace(-pi/2, pi/2, 30), linspace(pi/2, 0, 30)]
%     set(handle_axes, 'XLim', [-0.4,0.4], 'YLim', [-0.2,0.4], 'ZLim', [0,0.4]);
%     trf_q3= makehgtform('axisrotate', j5_rot_axis_j5, q5);
%     set(trf_link3_joint3, 'Matrix', trf_q3);
%     drawnow;
%     pause(0.02);
% end
%% 

% Animation: All joints together.
q_init= 0.5*ones(4,1); % This leads to all joints being at 0.

for i= 1:20
    q_next= rand(4,1); 
    % rand() gives uniformly distributed random numbers in the interval [0,1]
    
    for t=0:0.02:1
        q= q_init + t*(q_next - q_init);
        q1= (pi/2)*(2*q(1) - 1);
        q2= (pi/2)*(2*q(2) - 1);
        q3= (pi/2)*(2*q(3) - 1);
        a4= (0.04)*(2*q(4) - 1);
        
        set(handle_axes, 'XLim', [-0.4,0.4], 'YLim', [-0.2,0.4], 'ZLim', [0,0.4]);
        trf_q1= makehgtform('axisrotate', j1_rot_axis_j1, q1);
        set(trf_link1_joint1, 'Matrix', trf_q1);
        
        set(handle_axes, 'XLim', [-0.4,0.4], 'YLim', [-0.2,0.4], 'ZLim', [0,0.4]);
        trf_q2= makehgtform('axisrotate', j2_rot_axis_j2, q2);
        set(trf_link2_joint2, 'Matrix', trf_q2);
        
        set(handle_axes, 'XLim', [-0.4,0.4], 'YLim', [-0.2,0.4], 'ZLim', [0,0.4]);
        trf_q3= makehgtform('axisrotate', j3_rot_axis_j3, q3);
        set(trf_link3_joint3, 'Matrix', trf_q3);
        
        set(handle_axes, 'XLim', [-0.4,0.4], 'YLim', [-0.2,0.4], 'ZLim', [0,0.4]);
        trf_a4= makehgtform('translate', j4_translation_axis_j4*a4);
        set(trf_link4_joint4, 'Matrix', trf_a4);
        drawnow;
        pause(0.005);
        
    end
    
    q_init= q_next;
end
    