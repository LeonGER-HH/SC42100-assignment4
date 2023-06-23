addpath("helpers")
% initialize
nx = size(A1,2);
nu = size(B1,2);

Np = 5; %prediction horizon
alpha = 0.003; %convergence rate
Tfinal=5;
umax=60;

A_lst = {A1, A2, A3, A4};
B_lst = {B1, B2, B3, B4};
x0_lst = {x01, x02, x03, x04};
%% running the centralized optimization
u1 = sdpvar(2*Np, 1);
u2 = sdpvar(2*Np, 1);
u3 = sdpvar(2*Np, 1);
u4 = sdpvar(2*Np, 1);
u_lst = {u1, u2, u3, u4};
xf = sdpvar(4,1);
cost = 0;
const = [];
[P1, S1] = get_predMat(A1, B1, Np);
[P2, S2] = get_predMat(A2, B2, Np);
[P3, S3] = get_predMat(A3, B3, Np);
[P4, S4] = get_predMat(A4, B4, Np);

for i=1:4
    A = A_lst{i};
    B = B_lst{i};
    x0 = x0_lst{i};
    [P, S] = get_predMat(A, B, Np);
    Pp = P(1:20,:); Sp = S(1:20,:);
    cost = cost + (Pp*x0)'*Pp*x0 + (Pp*x0)'*Sp*u_lst{i}+(Sp*u_lst{i})'*Pp*x0 + u_lst{i}'*(Sp'*Sp+eye(size(Sp,2)))*u_lst{i};
    const = [const;P(21:end,:)*x0+S(21:end,:)*u_lst{i}==xf; abs(u_lst{i})<= umax/Tfinal*ones(size(u1))];
    const = [const;P(21:end,:)*x0+S(21:end,:)*u_lst{i}==xf; u_lst{i}'*u_lst{i}<= umax^2/(Tfinal+1)];
end

opts = sdpsettings('solver', 'gurobi', 'verbose', 1);
result = optimize(const, cost,opts);
xf_cntr = value(xf)

%ground truth vectors
x_a1 = P1*x01 + S1*value(u1);
x_a1_gt = reshape(x_a1,[4,6]);
x_a2 = P2*x02 + S2*value(u2);
x_a2_gt = reshape(x_a2,[4,6]);
x_a3 = P3*x03 + S3*value(u3);
x_a3_gt = reshape(x_a3,[4,6]);
x_a4 = P4*x04 + S4*value(u4);
x_a4_gt = reshape(x_a4,[4,6]);

%% new try with quadratic programming directly in Matlab
[P1, S1] = get_predMat(A1, B1, Np);
[P2, S2] = get_predMat(A2, B2, Np);
[P3, S3] = get_predMat(A3, B3, Np);
[P4, S4] = get_predMat(A4, B4, Np);
H1 = 2*(S1(1:20,:)'*S1(1:20,:)+eye(2*Np));
H2 = 2*(S2(1:20,:)'*S2(1:20,:)+eye(2*Np));
H3 = 2*(S3(1:20,:)'*S3(1:20,:)+eye(2*Np));
H4 = 2*(S4(1:20,:)'*S4(1:20,:)+eye(2*Np));
lb_u = -1*umax/Tfinal*ones(2*Np, 1);
ub_u = umax/Tfinal*ones(2*Np, 1);

lambda12 = zeros(4,1);
lambda23 = zeros(4,1);
lambda34 = zeros(4,1);


xf_1=zeros(4,1);xf_2=zeros(4,1);xf_3=zeros(4,1);xf_4=zeros(4,1);

maxIters = 5000;
n = 1;
options = optimoptions('quadprog','Display','off');
xf_log = zeros(4,maxIters,4);
lambda_diff_log = zeros(1,maxIters+1);
mu = 0.9; %momentum term
theta = zeros(12,1); % initialize
alpha = 0.002;
while n <= maxIters
%     alpha = 0.005 - 0.004/maxIters*n;
    f1 = (2*x01'*P1(1:20,:)'*S1(1:20,:)+lambda12'*S1(21:end,:))';
    f2 = (2*x02'*P2(1:20,:)'*S2(1:20,:)+(lambda23-lambda12)'*S2(21:end,:))';
    f3 = (2*x03'*P3(1:20,:)'*S3(1:20,:)+(lambda34-lambda23)'*S3(21:end,:))';
    f4 = (2*x04'*P4(1:20,:)'*S4(1:20,:)-lambda34'*S4(21:end,:))';

    u1 = quadprog(H1,f1', [],[],[],[],lb_u, ub_u,[],options);
    xf_1 = P1(21:end,:)*x01+S1(21:end,:)*u1;
    u2 = quadprog(H2,f2', [],[],[],[],lb_u, ub_u,[],options);
    xf_2 = P2(21:end,:)*x02+S2(21:end,:)*u2;
    u3 = quadprog(H3,f3', [],[],[],[],lb_u, ub_u,[],options);
    xf_3 = P3(21:end,:)*x03+S3(21:end,:)*u3;
    u4 = quadprog(H4,f4', [],[],[],[],lb_u, ub_u,[],options);
    xf_4 = P4(21:end,:)*x04+S4(21:end,:)*u4;
    
    lambda12_old = lambda12;
    lambda23_old = lambda23;
    lambda34_old = lambda34;
    lambda12 = lambda12 + alpha*(xf_1-xf_2);
    lambda23 = lambda23 + alpha*(xf_2-xf_3);
    lambda34 = lambda34 + alpha*(xf_3-xf_4);

    %implementing the momentum accelerator
    theta_old = theta;
    theta = [lambda12; lambda23; lambda34];
    lambda = theta + mu*(theta-theta_old);
    lambda12 = lambda(1:4);
    lambda23 = lambda(5:8);
    lambda34 = lambda(9:12);
    %logging and printing during the loop
    xf_log(:,n,1) = xf_1; xf_log(:,n,2) = xf_2; xf_log(:,n,3) = xf_3;xf_log(:,n,4) = xf_4;
    lambda_diff_log(1,n) = norm(lambda12-lambda12_old)+norm(lambda23-lambda23_old)+norm(lambda34-lambda34_old);
    formatSpec = 'Iteration %4.1f : %8.4f \n';
    fprintf(formatSpec, n, lambda_diff_log(n));
    n = n+1;
%     pause(3)
end


%% plotting
% Q1: error sequence|x1_i - x1_centralized|
len = size(xf_log,2);
error_agent1 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,1),[4,len])));
error_agent2 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,2),[4,len])));
error_agent3 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,3),[4,len])));
error_agent4 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,4),[4,len])));
steps = 1:len;
[mu, q] = convergence_analysis(error_agent1, x_a1_gt(1,end));
q = q(end)
mu = mu(end)
figure(1)
plot(steps, error_agent1, steps, error_agent2, steps, error_agent3, steps, error_agent4)
ylim([-2 20])
ylabel("||x_f - x_{f, centralized}||_1")
xlabel("iteration step")
legend("aircraft 1","aircraft 2","aircraft 3","aircraft 4")
% saveas(gcf,'figures4report/q1-error_plot.png')
% saveas(gcf,'figures4report/q1b-alpha8-error_plot.png')
saveas(gcf,'figures4report/q1c-accelerated_error_plot.png')

%%
t = 0:5;
% Q1: state trajectories
x_a1 = P1*x01 + S1*value(u1);
x_a1 = reshape(x_a1,[4,6]);
x_a2 = P2*x02 + S2*value(u2);
x_a2 = reshape(x_a2,[4,6]);
x_a3 = P3*x03 + S3*value(u3);
x_a3 = reshape(x_a3,[4,6]);
x_a4 = P4*x04 + S4*value(u4);
x_a4 = reshape(x_a4,[4,6]);

figure(2)
plot(t, x_a1_gt(1,:),"r-.", t, x_a2_gt(1,:),"r--", t, x_a3_gt(1,:),"r--", t, x_a4_gt(1,:),"r--")
hold on
plot(t, x_a1(1,:),t, x_a2(1,:),t, x_a3(1,:),t, x_a4(1,:))
xlabel("time step t")
ylabel("state x1")
legend("centr. sol.", "", "", "", "aircraft 1","aircraft 2","aircraft 3","aircraft 4")
saveas(gcf,'figures4report/q1-state_trajectory_x1.png')

figure(3)
plot(t, x_a1_gt(2,:),"r-.", t, x_a2_gt(2,:),"r--", t, x_a3_gt(2,:),"r--", t, x_a4_gt(2,:),"r--")
hold on
plot(t, x_a1(2,:),t, x_a2(2,:),t, x_a3(2,:),t, x_a4(2,:))
xlabel("time step t")
ylabel("state x2")
legend("centr. sol.", "", "", "", "aircraft 1","aircraft 2","aircraft 3","aircraft 4")
saveas(gcf,'figures4report/q1-state_trajectory_x2.png')

figure(4)
plot(t, x_a1_gt(3,:),"r-.", t, x_a2_gt(3,:),"r--", t, x_a3_gt(3,:),"r--", t, x_a4_gt(3,:),"r--")
hold on
plot(t, x_a1(3,:),t, x_a2(3,:),t, x_a3(3,:),t, x_a4(3,:))
xlabel("time step t")
ylabel("state x3")
legend("centr. sol.", "", "", "", "aircraft 1","aircraft 2","aircraft 3","aircraft 4")
saveas(gcf,'figures4report/q1-state_trajectory_x3.png')

figure(5)
plot(t, x_a1_gt(4,:),"r-.", t, x_a2_gt(4,:),"r--", t, x_a3_gt(4,:),"r--", t, x_a4_gt(4,:),"r--")
hold on
plot(t, x_a1(4,:),t, x_a2(4,:),t, x_a3(4,:),t, x_a4(4,:))
xlabel("time step t")
ylabel("state x4")
legend("centr. sol.", "", "", "", "aircraft 1","aircraft 2","aircraft 3","aircraft 4")
saveas(gcf,'figures4report/q1-state_trajectory_x4.png')


%%

W = [0.75, 0.25, 0, 0;
    0.25, 0.5, 0.25, 0;
    0, 0.25, 0.5, 0.25;
    0, 0, 0.25, 0.75];





f1 = 2*x01'*P1(1:20,:)'*S1(1:20,:);
f2 = 2*x02'*P2(1:20,:)'*S2(1:20,:);
f3 = 2*x03'*P3(1:20,:)'*S3(1:20,:);
f4 = 2*x04'*P4(1:20,:)'*S4(1:20,:);

iterations = 70;
phi_error_log = zeros(4,iterations,5);

xf_norm_log = zeros(4,iterations);
alpha = 0.1;
phi =30;
options = optimoptions('quadprog','Display','off');
phis = [20,30,33,36,40];
for i=1:5
    phi = phis(1, i);
    xf_local_1 = x01;
    xf_local_2 = x02;
    xf_local_3 = x03;
    xf_local_4 = x04;
    for k=1:iterations
        W_now = W^phi;
        
        [u1,~,~,~,lambda1] = quadprog(H1,f1', [],[],S1(21:end,:),xf_local_1-P1(21:end,:)*x01,lb_u, ub_u,[],options);
        xf_local_1 = P1(21:end,:)*x01+S1(21:end,:)*u1;
        [u2,~,~,~,lambda2] = quadprog(H2,f2', [],[],S2(21:end,:),xf_local_2-P2(21:end,:)*x02,lb_u, ub_u,[],options);
        xf_local_2 = P2(21:end,:)*x02+S2(21:end,:)*u2;
        [u3,~,~,~,lambda3] = quadprog(H3,f3', [],[],S3(21:end,:),xf_local_3-P3(21:end,:)*x03,lb_u, ub_u,[],options);
        xf_local_3 = P3(21:end,:)*x03+S3(21:end,:)*u3;
        [u4,~,~,~,lambda4] = quadprog(H4,f4', [],[],S4(21:end,:),xf_local_4-P4(21:end,:)*x04,lb_u, ub_u,[],options);
        xf_local_4 = P4(21:end,:)*x04+S4(21:end,:)*u4;
    
        G = [xf_local_1 + alpha*lambda1.eqlin,...
            xf_local_2 + alpha*lambda2.eqlin,...
            xf_local_3 + alpha*lambda3.eqlin,...
            xf_local_4 + alpha*lambda4.eqlin];
    
        xf_local_1 = sum(W_now(1,:).*G, 2);
        xf_local_2 = sum(W_now(2,:).*G, 2);
        xf_local_3 = sum(W_now(3,:).*G, 2);
        xf_local_4 = sum(W_now(4,:).*G, 2);
    %     xf_log(:,k,1) = xf_local_1; xf_log(:,k,2) = xf_local_2; xf_log(:,k,3) = xf_local_3;xf_log(:,k,4) = xf_local_4;
        phi_error_log(:,k,i) = xf_local_1;
        xf_norm_log(:,k) = sum([xf_local_1, xf_local_2, xf_local_3, xf_local_4],2).*0.25;
    end
end

% plot(xf_norm_log')
% xf_local_1
% plot(xf_norm_log(1,:))

%% plotting
% Q1d: error sequence|x1_i - x1_centralized|
len = size(phi_error_log,2);
phi1 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(phi_error_log(:,:,1),[4,len])));
phi2 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(phi_error_log(:,:,2),[4,len])));
phi3 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(phi_error_log(:,:,3),[4,len])));
phi4 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(phi_error_log(:,:,4),[4,len])));
phi5 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(phi_error_log(:,:,5),[4,len])));

steps = 1:len;
disp(['phi1=', phis(1,1)])
[mu, q] = convergence_analysis(phi1, zeros(4,1));
q = q(end)
mu = mu(end)
disp(['phi2=', phis(1,2)])
[mu, q] = convergence_analysis(phi2, zeros(1,1));
q = q(end)
mu = mu(end)
disp(['phi3=', phis(1,3)])
[mu, q] = convergence_analysis(phi3, zeros(4,1));
q = q(end)
mu = mu(end)
disp(['phi4=', phis(1,4)])
[mu, q] = convergence_analysis(phi4, zeros(4,1));
q = q(end)
mu = mu(end)
disp(['phi5=', phis(1,5)])
[mu, q] = convergence_analysis(phi5, zeros(4,1));
q = q(end)
mu = mu(end)
figure(1)
plot(steps, phi1, steps, phi2, steps, phi3, steps, phi4, steps, phi5)
ylim([-2 20])
ylabel("||x_f - x_{f, centralized}||_1")
xlabel("iteration step")
legend("\phi=10","\phi=20","\phi=30","\phi=40", "\phi=50")

saveas(gcf,'figures4report/q1d-consensus.png')


%% ADMM with Consensus
iterations = 100;
rho = 3;
rhos = [0.1,0.5,0.7,1,3,10];
% rho = 0.7
    xf_log = zeros(4,iterations,length(rhos));


for rho_i=1:length(rhos)
    rho = rhos(rho_i)
    Pp1 = P1(1:20,:);Pp2 = P2(1:20,:);Pp3 = P3(1:20,:);Pp4 = P4(1:20,:);
    Sp1 = S1(1:20,:);Sp2 = S2(1:20,:);Sp3 = S3(1:20,:);Sp4 = S4(1:20,:);
    Pf1 = P1(21:end,:);Pf2 = P2(21:end,:);Pf3 = P3(21:end,:);Pf4 = P4(21:end,:);
    Sf1 = S1(21:end,:);Sf2 = S2(21:end,:);Sf3 = S3(21:end,:);Sf4 = S4(21:end,:);
    
    %optimization problem's quadratic term
    H1 = 2*(Sp1'*Sp1 +eye(2*Np)+rho/2*(Sf1'*Sf1));
    H2 = 2*(Sp2'*Sp2 +eye(2*Np)+rho/2*(Sf2'*Sf2));
    H3 = 2*(Sp3'*Sp3 +eye(2*Np)+rho/2*(Sf3'*Sf3));
    H4 = 2*(Sp4'*Sp4 +eye(2*Np)+rho/2*(Sf4'*Sf4));
    xf_local_1 = x01;
    xf_local_2 = x02;
    xf_local_3 = x03;
    xf_local_4 = x04;
    lambda1 = zeros(4,iterations+1);
    lambda2 = zeros(4,iterations+1);
    lambda3 = zeros(4,iterations+1);
    lambda4 = zeros(4,iterations+1);
    z = zeros(4,iterations+1); %initialize global variable

    for k=1:iterations
        f1 = 2*x01'*Pp1'*Sp1+rho/2*(x01'*Pf1'*Sf1-z(:,k)'*Sf1+(Pf1*x01-z(:,k))'*Sf1)+lambda1(:,k)'*Sf1;
        f2 = 2*x02'*Pp2'*Sp2+rho/2*(x02'*Pf2'*Sf2-z(:,k)'*Sf2+(Pf2*x02-z(:,k))'*Sf2)+lambda2(:,k)'*Sf2;
        f3 = 2*x03'*Pp3'*Sp3+rho/2*(x03'*Pf3'*Sf3-z(:,k)'*Sf3+(Pf3*x03-z(:,k))'*Sf3)+lambda3(:,k)'*Sf3;
        f4 = 2*x04'*Pp4'*Sp4+rho/2*(x04'*Pf4'*Sf4-z(:,k)'*Sf4+(Pf4*x04-z(:,k))'*Sf4)+lambda4(:,k)'*Sf4;
    
        u1 = quadprog(H1,f1', [],[],[],[],lb_u, ub_u,[],options);
        xf_local_1 = P1(21:end,:)*x01+S1(21:end,:)*u1;
        u2 = quadprog(H2,f2', [],[],[],[],lb_u, ub_u,[],options);
        xf_local_2 = P2(21:end,:)*x02+S2(21:end,:)*u2;
        u3 = quadprog(H3,f3', [],[],[],[],lb_u, ub_u,[],options);
        xf_local_3 = P3(21:end,:)*x03+S3(21:end,:)*u3;
        u4 = quadprog(H4,f4', [],[],[],[],lb_u, ub_u,[],options);
        xf_local_4 = P4(21:end,:)*x04+S4(21:end,:)*u4;
    
        z(:,k+1) = 1/4 * (xf_local_1 + lambda1(:,k)/rho +...
                      xf_local_2 + lambda2(:,k)/rho+...
                      xf_local_3 + lambda3(:,k)/rho+...
                        xf_local_4 + lambda4(:,k)/rho);
        lambda1(:,k+1) = lambda1(:,k) + rho*(xf_local_1 - z(:,k+1));
        lambda2(:,k+1) = lambda2(:,k) + rho*(xf_local_2 - z(:,k+1));
        lambda3(:,k+1) = lambda3(:,k) + rho*(xf_local_3 - z(:,k+1));
        lambda4(:,k+1) = lambda4(:,k) + rho*(xf_local_4 - z(:,k+1));
%         xf_log(:,k,1) = xf_local_1; xf_log(:,k,2) = xf_local_2; xf_log(:,k,3) = xf_local_3;xf_log(:,k,4) = xf_local_4;
        xf_log(:,k,rho_i) = 1/4*(xf_local_1+xf_local_2+xf_local_3+xf_local_4);
    end
end
xf_local_1
xf_local_2
%% plotting error sequence for different rho
len = size(xf_log,2);
rho1 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,1),[4,len])));
rho2 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,2),[4,len])));
rho3 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,3),[4,len])));
rho4 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,4),[4,len])));
rho5 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,5),[4,len])));
rho6 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,6),[4,len])));

% plot(rho1, rho2, rho3, rho4, rho5, rho6)
steps = 1:len;
plot(steps,rho1,steps,rho2,steps,rho3,steps,rho4,steps,rho5,steps,rho6)
ylabel("||x_f - x_{f, centralized}||_1")
xlabel("iteration step")
% rhos = [0.1,0.5,0.7,1,3,10];
legend("\rho=0.1","\rho=0.5","\rho=0.7","\rho=1","\rho=3","\rho=10")
saveas(gcf,'figures4report/q2-rho_influence.png')

%% plotting
% Q2: error sequence|x1_i - x1_centralized|
len = size(xf_log,2);
error_agent1 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,1),[4,len])));
error_agent2 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,2),[4,len])));
error_agent3 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,3),[4,len])));
error_agent4 = sum(abs(xf_cntr(:)*ones(1,len) - reshape(xf_log(:,:,4),[4,len])));
steps = 1:len;
[mu, q] = convergence_analysis(error_agent1, 0, 1);

figure(1)
plot(steps, error_agent1, steps, error_agent2, steps, error_agent3, steps, error_agent4)
ylim([-2 20])
ylabel("||x_f - x_{f, centralized}||_1")
xlabel("iteration step")
legend("aircraft 1","aircraft 2","aircraft 3","aircraft 4")
% saveas(gcf,'figures4report/q1-error_plot.png')
% saveas(gcf,'figures4report/q1b-alpha8-error_plot.png')
saveas(gcf,'figures4report/q2-ADMM_consensus.png')
%%
[mu, q] = convergence_analysis(error_agent1, 0, 0);
plot(q)
xlabel("iteration step")
ylabel("order of convergence q")
saveas(gcf,'figures4report/q2b_q_noise.png')
%%
t = 0:5;
% Q1: state trajectories
x_a1 = P1*x01 + S1*value(u1);
x_a1 = reshape(x_a1,[4,6]);
x_a2 = P2*x02 + S2*value(u2);
x_a2 = reshape(x_a2,[4,6]);
x_a3 = P3*x03 + S3*value(u3);
x_a3 = reshape(x_a3,[4,6]);
x_a4 = P4*x04 + S4*value(u4);
x_a4 = reshape(x_a4,[4,6]);

figure(2)
plot(t, x_a1_gt(1,:),"r-.", t, x_a2_gt(1,:),"r--", t, x_a3_gt(1,:),"r--", t, x_a4_gt(1,:),"r--")
hold on
plot(t, x_a1(1,:),t, x_a2(1,:),t, x_a3(1,:),t, x_a4(1,:))
xlabel("time step t")
ylabel("state x1")
legend("centr. sol.", "", "", "", "aircraft 1","aircraft 2","aircraft 3","aircraft 4")
saveas(gcf,'figures4report/q2-state_trajectory_x1.png')

figure(3)
plot(t, x_a1_gt(2,:),"r-.", t, x_a2_gt(2,:),"r--", t, x_a3_gt(2,:),"r--", t, x_a4_gt(2,:),"r--")
hold on
plot(t, x_a1(2,:),t, x_a2(2,:),t, x_a3(2,:),t, x_a4(2,:))
xlabel("time step t")
ylabel("state x2")
legend("centr. sol.", "", "", "", "aircraft 1","aircraft 2","aircraft 3","aircraft 4")
saveas(gcf,'figures4report/q2-state_trajectory_x2.png')

figure(4)
plot(t, x_a1_gt(3,:),"r-.", t, x_a2_gt(3,:),"r--", t, x_a3_gt(3,:),"r--", t, x_a4_gt(3,:),"r--")
hold on
plot(t, x_a1(3,:),t, x_a2(3,:),t, x_a3(3,:),t, x_a4(3,:))
xlabel("time step t")
ylabel("state x3")
legend("centr. sol.", "", "", "", "aircraft 1","aircraft 2","aircraft 3","aircraft 4")
saveas(gcf,'figures4report/q2-state_trajectory_x3.png')

figure(5)
plot(t, x_a1_gt(4,:),"r-.", t, x_a2_gt(4,:),"r--", t, x_a3_gt(4,:),"r--", t, x_a4_gt(4,:),"r--")
hold on
plot(t, x_a1(4,:),t, x_a2(4,:),t, x_a3(4,:),t, x_a4(4,:))
xlabel("time step t")
ylabel("state x4")
legend("centr. sol.", "", "", "", "aircraft 1","aircraft 2","aircraft 3","aircraft 4")
saveas(gcf,'figures4report/q2-state_trajectory_x4.png')