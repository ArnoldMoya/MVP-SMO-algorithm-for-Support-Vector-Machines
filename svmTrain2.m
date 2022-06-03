function [model] = svmTrain2(X, Y, C, kernelFunction, ...
                            tol, max_passes)
%SVMTRAIN Trains an SVM classifier using MVP SMO algorithm. 
global kS D
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-3;
end

if ~exist('max_passes', 'var') || isempty(max_passes)
    max_passes = 5;
end

% Data parameters
m = size(X, 1);
n = size(X, 2);

% Map 0 to -1
Y(Y==0) = -1;

% Variables
alphas = zeros(m, 1);
b = 0;
E = zeros(m, 1);
passes = 0;
eta = 0;
L = 0;
H = 0;


% Pre-compute the Kernel Matrix
% The following can be slow due to the lack of vectorization
K = zeros(m);
for i = 1:m
    for j = i:m
         K(i,j) = kernelFunction(X(i,:)', X(j,:)');
         K(j,i) = K(i,j); %the matrix is symmetric
    end
end

% Q matrix
Q = zeros(m);
for i = 1:m
    for j = i:m
         Q(i,j) = Y(i)*Y(j)*kernelFunction(X(i,:)', X(j,:)');
         Q(j,i) = Q(i,j); %the matrix is symmetric
    end
end
    
% Train
fprintf('\nTraining ...');
dots = 12;
while passes < max_passes,
            
    num_changed_alphas = 0;
    %i,j are choosen by MVP heuristic
    Ii = ((Y == 1).*(alphas>0)) + ((Y == -1).*(alphas<C)); %(32)
    Ij = ((Y == 1).*(alphas<C)) + ((Y == -1).*(alphas>0)); %(33)

    tIi = (Y.* (Q*alphas - 1)).*Ii;
    i = find(tIi == max(tIi),1);%(35)

    tIj = (Y.* (Q*alphas - 1)).*Ij;
    j = find(tIj == min(tIj),1);%(36)

    % Calculate Ei = f(x(i)) - y(i) using (2) from [10]. 
    E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);

    if ((Y(i)*E(i) < -tol && alphas(i) < C) || ...
            (Y(i)*E(i) > tol && alphas(i) > 0))

        % Calculate Ej = f(x(j)) - y(j) using (44).
        E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);

        % Save old alphas
        alpha_i_old = alphas(i);
        alpha_j_old = alphas(j);

        % Compute L and H by (40) or (41). 
        if (Y(i) == Y(j)),
            L = max(0, alphas(j) + alphas(i) - C);
            H = min(C, alphas(j) + alphas(i));
        else
            L = max(0, alphas(j) - alphas(i));
            H = min(C, C + alphas(j) - alphas(i));
        end

        if (L == H),
            % continue to next i. 
            continue;
        end

        % Compute eta by (45).
        eta = 2 * K(i,j) - K(i,i) - K(j,j);
        if (eta >= 0),
            % continue to next i. 
            continue;
        end

        % Compute and clip new value for alpha j using (43) and (46).
        alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;

        % Clip
        alphas(j) = min (H, alphas(j));
        alphas(j) = max (L, alphas(j));

        % Check if change in alpha is significant
        if (abs(alphas(j) - alpha_j_old) < tol),
            % continue to next i. 
            % replace anyway
            alphas(j) = alpha_j_old;
            continue;
        end

        % Determine value for alpha i using (47). 
        alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));

        % Compute b1 and b2 using (49) and (50) respectively. 
        b1 = b - E(i) ...
             - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
             - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
        b2 = b - E(j) ...
             - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
             - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';

        % Compute b by (19). 
        if (0 < alphas(i) && alphas(i) < C),
            b = b1;
        elseif (0 < alphas(j) && alphas(j) < C),
            b = b2;
        else
            b = (b1+b2)/2;
        end


        kS = [kS, max(kS)+1];
        D = [D, 0.5*alphas'*Q*alphas - sum(alphas)];

        num_changed_alphas = num_changed_alphas + 1;

        
    end
    
    if (num_changed_alphas == 0),
        passes = passes + 1;
    else
        passes = 0;
    end

    fprintf('.');
    dots = dots + 1;
    if dots > 78
        dots = 0;
        fprintf('\n');
    end

end
fprintf(' Done! \n\n');

% Save the model
idx = alphas > 0;
model.X= X(idx,:);
model.y= Y(idx);
model.Q = Q;
model.K = K;
model.kernelFunction = kernelFunction;
model.b= b;
model.alphas= alphas(idx);
model.a = alphas;
model.w = ((alphas.*Y)'*X)';

end
