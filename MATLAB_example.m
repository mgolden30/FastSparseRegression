%{
MATLAB example code for using SPRINT in both variants
%}

clear;

addpath("MATLAB/");

A = magic(100); %Make a magic square
A = A-mean(A, "all");

%Find approximate sparse null vectors of this matrix in three ways
tiledlayout(2,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1: exhaustive (slow) takes 1.4 seconds on my machine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tic
[c1, residuals1] = exhaustive_search(A);
toc

nexttile
%plot nonzero coefficients
imagesc( c1 ~= 0 );
title("exhaustive search coefficients");
axis square

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2: SPRINT_minus (fast) 0.39 seconds on my machine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tic
[c2, residuals2] = SPRINT_minus(A);
toc

nexttile
%plot nonzero coefficients
imagesc( c2 ~= 0 );
title("SPRINT- coefficients");
axis square

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2: SPRINT_plus (fastest) 0.18 seconds on my machine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tic
%seed with one-term model from SPRINT_minus
c0 = c2(:,1);
n_max = 50; %only go up to 50 terms
[c3, residuals3] = SPRINT_plus( A, c0, n_max );
toc

nexttile
%plot nonzero coefficients
imagesc( c3 ~= 0 );
title("SPRINT+ coefficients");
axis square

%%Lastly, compare residuals
nexttile
semilogy( 1:100, residuals1 );
hold on
semilogy( 1:100, residuals2 );
semilogy( 1:100, residuals3 );
hold off
legend({"exhaustive", "SPRINT-", "SPRINT+"});
title("residuals");
xlabel("terms in model");
ylabel("residual");