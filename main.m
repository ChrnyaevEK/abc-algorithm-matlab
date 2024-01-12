clc; clear;

p_eBeePopSize = 10;
p_oBeePopSize = 10;
p_trialLim = 1;
p_dim = 10;
p_lim = [-65.536, 65.536];
p_epochs = 100;
p_tolerance = 0.005;
p_patience = p_epochs;
p_visualize = false;

[x, y, history] = abc(@rastrigin, p_eBeePopSize, p_oBeePopSize, p_trialLim, p_dim, p_lim, p_epochs, p_tolerance, p_patience, p_visualize);
plot(history);
% 
% stats = zeros(100, 1);
% statsTime = zeros(100, 1);
% 
% f = figure(1);
% 
% ax = subplot(1, 3, 1);
% title('Cost progress');
% xlabel('Epoch');
% ylabel('Cost');
% 
% hold on;
% for attempt = 1:100
%     start = tic;
%     [x, y, history] = abc(@dejong, p_eBeePopSize, p_oBeePopSize, p_trialLim, p_dim, p_lim, p_epochs, p_tolerance, p_patience, p_visualize);
%     finish = toc;
% 
%     stats(attempt) = y;
%     statsTime(attempt) = finish;
% 
%     plot(history);
% end
% hold off;
% 
% subplot(1, 3, 2);
% histogram(stats, 20);
% title('Final value distribution');
% xlabel('Cost');
% 
% subplot(1, 3, 3);
% histogram(statsTime, 20);
% title('Evaluation time');
% xlabel('Time [s]');
% 
% figure(2);
% clf;
% plot(history);

function [x, y, history] = abc(fn, eBeePopSize, oBeePopSize, trialLim, dim, lim, epochs, tolerance, patience, visualize)
    % eBeePopSize - number of empoyed bees
    % oBeePopSize - number of onlooker bees
    % trialLim    - number of attempts to improve solution
    % dim       - number of parameters (dimentions)
    % lim       - range of parameter values
    % epochs    - max generations allowed
    % tolerance - required minimal change in objective function for `patience` epochs
    % patience  - wait for change in cost function for N epoch
    % visualize - enable / disable visualization
    
    assert(eBeePopSize > 0);
    assert(oBeePopSize > 0);
    assert(trialLim >= 1);
    assert(dim > 1);  % Due to visualization
    assert(length(lim) == 2);
    assert(epochs > 0);
    assert(patience > 0);
    assert(tolerance > 0);

    % Termination logic
    terminate = @(dy) abs(dy) < tolerance;
 
    bestY = inf;
    bestYOld = inf;
    bestX = nan;

    history = nan(epochs, 1);

    % Initial population of food sources
    foodSources = lim(1) + rand(eBeePopSize, dim) * (lim(2) - lim(1));
    index = zeros(eBeePopSize, 2);  % [cost, trial]

    index(:, 1) = evalCost(foodSources, fn);

    count = patience;
    if visualize
        figure(1);
        clf;
    end
    for epoch=1:epochs
        fprintf('Epoch: %i, best: %.2f, patience: %i\n', epoch, bestY, count);
        % Draw epoch start solution
        if visualize
            % clf;
            title('Bee (Jees) artificial colony (PCA soluion representation)');
            xlabel('PC1');
            ylabel('PC2');
            % [~, pos] = min(index(:,1));
            % [plt1, pltBest1] = draw(foodSources, "#0072BD", pos);
        end
        % Employed bee phase
        fs = foodSources(:,:);
        for eBee = 1 : eBeePopSize
            % Pick food source and partner
            own = foodSources(eBee, :);
            partner = foodSources(randi(eBeePopSize),:);
            
            var = randi(dim);  % Select which variable to change
            phi = -1 + rand() * 2;  % Random number from [-1, 1]

            newFoodSource = own(:,:);
            newFoodSource(var) = own(var) + phi * (own(var) - partner(var));

            gCost = fn(newFoodSource);
            if gCost < index(eBee, 1)
                % Update solution
                index(eBee, 1) = gCost;
                index(eBee, 2) = 0;
                fs(eBee,:) = newFoodSource;
            else
                % Preserve old solution
                index(eBee, 2) = index(eBee, 2) + 1;
            end
        end
        % Draw employed bees phase results
        if visualize
            hold on;
            [~, pos] = min(index(:,1));
            [plt2, pltBest2] = draw(fs, "#4DBEEE", pos, 'o');
        end
        foodSources = fs(:,:);

        p = cumsum(index(:, 1) / sum(index(:, 1)));
        fs = foodSources(:,:);
        for oBee = 1: oBeePopSize
            selectedFs = find(p >= rand(), 1, 'first');

            own = foodSources(selectedFs,:);
            partner = foodSources(randi(eBeePopSize),:);
            
            var = randi(dim);  % Select which variable to change
            phi = -1 + rand() * 2;  % Random number from [-1, 1]

            newFoodSource = own(:);
            newFoodSource(var) = own(var) + phi * (own(var) - partner(var));

            gCost = fn(newFoodSource);
            if gCost < index(selectedFs, 1)
                % Update solution
                index(selectedFs, 1) = gCost;
                index(selectedFs, 2) = 0;
                fs(selectedFs,:) = newFoodSource;
            else
                % Preserve old solution
                index(selectedFs, 2) = index(selectedFs, 2) + 1;
            end
        end
        foodSources = fs(:,:);
        % Draw onlooker results
        if visualize
            hold on;
            [~, pos] = min(index(:,1));
            [plt3, ~] = draw(foodSources, "#4DBEEE", pos, 'square');
        end

        [tmpY, tmpYp] = min(index(:, 1));
        if tmpY <= bestY
            bestY = tmpY;
            bestX = foodSources(tmpYp,:);
        end
        history(epoch) = bestY;

        if terminate(bestYOld - bestY) || epoch == epochs
            count = count - 1;
            
            x = bestX;
            y = bestY;
            
            if ~count || epoch == epochs
                fprintf('Done.\n');
                history = history(~(isnan(history))); 
                fprintf('Best cost: %.2f\nx: ', y);
                disp(x);
                return 
            end
        else
            % Big change detected - reset patience
            count = patience;
        end
        bestYOld = bestY;

        % Scout phase
        overLim = find(index(:, 2) >= trialLim);
        for sBee = 1:length(overLim)
            foodSources(overLim(sBee), :) = lim(1) + rand(1, dim) * (lim(2) - lim(1));
            index(overLim(sBee), 1) = fn(foodSources(sBee, :));
            index(overLim(sBee), 2) = 0;
        end
        % Draw scouter results
        if visualize
            hold on;
            [~, pos] = min(index(:,1));
            [plt4, ~] = draw(foodSources, "#4DBEEE", pos, '^');
        end
        if epoch == 1 && visualize
            legend([plt2, plt3, plt4, pltBest2], {'Employed', 'On looker', 'Scouter', 'Epoch best'}, 'AutoUpdate','off')
        end
    end

    function h = evalCost(pop, fn)
        h = zeros(size(pop, 1), 1);
        for i = 1:size(pop, 1)
           h(i) = fn(pop(i, :));
        end
    end

    function [plt, pltBest] = draw(data, color, pos, mrk)
        [~, score, ~] = pca(data);
        % Reduce dimensionality to 2
        reduced = score(:, 1:2);
        % Plot the original and reduced data
        hold on;
        plt = scatter(reduced(:, 1), reduced(:, 2), 'filled', mrk, 'MarkerFaceColor', color);
        hold on;
        pltBest = scatter(reduced(pos, 1), reduced(pos, 2), 'filled', 'pentagram', 'MarkerFaceColor', 'r');
        % pause(0.1);
    end
end



