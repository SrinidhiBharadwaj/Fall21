clc
clear
close all

% Load training set 

TrainingSet = load("TrainingSamplesDCT_8_new.mat");
foreground = TrainingSet.TrainsampleDCT_FG;
background = TrainingSet.TrainsampleDCT_BG;

% ---------------------------------------------------------------------- %
% Problem a

[fg_rows, fg_cols] = size(foreground);
[bg_rows, bg_cols] = size(background);

total_samples = fg_rows + bg_rows;

prior_foreground = fg_rows / total_samples;
prior_background = bg_rows / total_samples;


% ---------------------------------------------------------------------- %
% Problem b

% Foreground and background means %
mu_foreground = mean(foreground);
mu_background = mean(background);

% Foreground and background variances %
var_foreground = var(foreground);
var_background = var(background);

% Foreground and background covariances %
covar_foreground = cov(foreground);
covar_background = cov(background);
wi_foreground = inv(covar_foreground);
wi_background = inv(covar_background);

z_distance = [];
for idx=1:fg_cols
    subplot(8,8,idx)
    set(gcf,'Position', get(0, 'ScreenSize'))

    fg_average = mean(foreground(:,idx));
    fg_variance = var(foreground(:,idx));
    fg_standard_deviation = sqrt(fg_variance);

    bg_average = mean(background(:,idx));
    bg_variance = var(background(:,idx));
    bg_standard_deviation = sqrt(bg_variance);

    % Considering 4 standard deviations for the Gaussian
    fore = fg_average-4*fg_standard_deviation:fg_standard_deviation/50:...
                        fg_average+4*fg_standard_deviation;
    back = bg_average-4*bg_standard_deviation:bg_standard_deviation/50:...
                        bg_average+4*bg_standard_deviation;
    x = sort([fore back]); % x-axis
    p_x_y_cheetah = gaussian(x,fg_average,fg_standard_deviation);
    p_x_y_grass = gaussian(x,bg_average,bg_standard_deviation);
    
    plot(x,p_x_y_cheetah,x,p_x_y_grass)
    title(['Index = ' num2str(idx)],'FontSize',8);
 
    % z-test for comparision
    z_tmp = z_test(fg_variance, fg_average, bg_variance, bg_average);
    z_distance = [z_distance z_tmp];
end

[z_sorted, z_idx] = sort(z_distance);
worstidx = sort(z_idx(1:8)); % Worst Index values
bestidx = sort(z_idx(57:64)); % Best Index values

figure();
for i=1:8
    subplot(2,8,i)
    worst_idx = worstidx(i);
    set(gcf,'Position', get(0, 'ScreenSize'))
    fg_average = mean(foreground(:,worst_idx));
    fg_variance = var(foreground(:,worst_idx));
    fg_standard_deviation = sqrt(fg_variance);

    bg_average = mean(background(:,worst_idx));
    bg_variance = var(background(:,worst_idx));
    bg_standard_deviation = sqrt(bg_variance);
    % Considering 4 standard deviations for the Gaussian
    fore = fg_average-4*fg_standard_deviation:fg_standard_deviation/50:...
                        fg_average+4*fg_standard_deviation;
    back = bg_average-4*bg_standard_deviation:bg_standard_deviation/50:...
                        bg_average+4*bg_standard_deviation;
    x = sort([fore back]); % x-axis
    p_x_y_cheetah = gaussian(x,fg_average,fg_standard_deviation);
    p_x_y_grass = gaussian(x,bg_average,bg_standard_deviation);
    
    plot(x,p_x_y_cheetah,x,p_x_y_grass)
    title(['Worst Index = ' num2str(worst_idx)],'FontSize',8);
end

for i=1:8
    subplot(2,8,i+8)
    %w = get(0, 'ScreenSize');
    best_idx = bestidx(i);
    set(gcf,'Position', get(0, 'ScreenSize'))
    fg_average = mean(foreground(:,best_idx));
    fg_variance = var(foreground(:,best_idx));
    fg_standard_deviation = sqrt(fg_variance);

    bg_average = mean(background(:,best_idx));
    bg_variance = var(background(:,best_idx));
    bg_standard_deviation = sqrt(bg_variance);
    % Considering 4 standard deviations for the Gaussian
    fore = fg_average-4*fg_standard_deviation:fg_standard_deviation/50:...
                        fg_average+4*fg_standard_deviation;
    back = bg_average-4*bg_standard_deviation:bg_standard_deviation/50:...
                        bg_average+4*bg_standard_deviation;
    x = sort([fore back]); % x-axis
    p_x_y_cheetah = gaussian(x,fg_average,fg_standard_deviation);
    p_x_y_grass = gaussian(x,bg_average,bg_standard_deviation);
    
    plot(x,p_x_y_cheetah,x,p_x_y_grass)
    title(['Best Index = ' num2str(best_idx)],'FontSize',8);
end

% ---------------------------------------------------------------------- %
% Problem c

pad_value = 7;
cheetah = imread("../cheetah.bmp");
cheetah = padarray(cheetah,[pad_value pad_value], 'post');
cheetah = im2double(cheetah);

[rows, cols] = size(cheetah);

unpadded_rows = rows-pad_value;
unpadded_cols = cols - pad_value;
converted_block = zeros(unpadded_rows, unpadded_cols);
zigzag = load("../Zig-Zag Pattern.txt");
zigzag = zigzag + 1; %Following MATLAB index
result_64 = zeros(unpadded_rows, unpadded_cols);
mu_foreground = transpose(mu_foreground);
mu_background = transpose(mu_background);
determinant_foreground = det(covar_foreground);
determinant_background = det(covar_background);

% Prediction with 64 dimensional gaussians
for i = 1:unpadded_rows
    for j = 1:unpadded_cols
        block = cheetah(i:i+7, j:j+7);
        transform = dct2(block); 
        % Create zigzag pattern
        zigzag_transform(zigzag) = transform;
        x = transpose(zigzag_transform);

        % Cheetah
        exp_func = -0.5 * transpose(x-mu_foreground) * wi_foreground * (x-mu_foreground);
        exp_coeff = 1/(sqrt(((2*pi)^64) * determinant_foreground));
        p_y_x_cheetah = exp_coeff * exp(exp_func) * prior_foreground;

        % Grass
        exp_func = -0.5 * transpose(x-mu_background) * wi_background * (x-mu_background);
        exp_coeff = 1/(sqrt(((2*pi)^64) * determinant_background));
        p_y_x_grass = exp_coeff * exp(exp_func) * prior_background;

        if(p_y_x_cheetah > p_y_x_grass)
            result_64(i, j) = 1;
        end
    end
end

result_8 = zeros(unpadded_rows, unpadded_cols);
mu_foreground_best = transpose(mean(foreground(:, bestidx)));
mu_background_best = transpose(mean(background(:, bestidx)));

cov_foreground_best = cov(foreground(:, bestidx));
cov_background_best = cov(background(:, bestidx));

wi_foreground_best = inv(cov_foreground_best);
wi_background_best = inv(cov_background_best);

determinant_best_fg = det(cov_foreground_best);
determinant_best_bg = det(cov_background_best);

% Prediction with 8 dimensional gaussians
for i = 1:unpadded_rows
    for j = 1:unpadded_cols
        block = cheetah(i:i+7, j:j+7);
        transform = dct2(block); % Utilize inbuilt MATLAB function for DCT calculation
        % Create zigzag pattern
        zigzag_transform(zigzag) = transform;
        x = transpose(zigzag_transform(:, bestidx));
        
        % Cheetah
        exp_func = -0.5 * transpose(x-mu_foreground_best) * wi_foreground_best ...
                                                  * (x-mu_foreground_best);
        exp_coeff = 1/(sqrt(((2*pi)^8) * determinant_best_fg));
        p_y_x_cheetah = exp_coeff * exp(exp_func) * prior_foreground;

        % Grass
        exp_func = -0.5 * transpose(x-mu_background_best) * wi_background_best ...
                                                  * (x-mu_background_best);
        exp_coeff = 1/(sqrt(((2*pi)^8) * determinant_best_bg));
        p_y_x_grass = exp_coeff * exp(exp_func) * prior_background;
        if(p_y_x_cheetah > p_y_x_grass)
            result_8(i, j) = 1;
        end
    end
end

figure;
imagesc(result_64);
title('Prediction with 64 dimensions')
colormap(gray(255));

figure;
imagesc(result_8);
title('Prediction with 8 dimensions')
colormap(gray(255));

% Error calculation

true_image = imread('../cheetah_mask.bmp');
error_64_dim = calc_error(true_image, result_64, prior_foreground, prior_background);
X = ['Probability of error for mask obtained using 64 dimensional Gaussian ' ...
            'is: ', num2str(error_64_dim), '%'];
disp(X)

error_8_dim = calc_error(true_image, result_8, prior_foreground, prior_background);
X = ['Probability of error for mask obtained using 8 dimensional Gaussian ' ...
            'is: ', num2str(error_8_dim), '%'];
disp(X)

% ---------------------------------------------------------------------- %
% Function to calculate the error
function error = calc_error(ref_image, actual_image, p_fore, p_back)
    foreground_pixels = 0;
    background_pixels = 0;
    
    for i=1:size(ref_image, 1)
        for j=1:size(ref_image, 2)
            if ref_image(i, j) == 255
                foreground_pixels = foreground_pixels + 1;
            else
                background_pixels = background_pixels + 1;
            end
        end
    end
    
    foreground_error = 0;
    background_error = 0;
    for i=1:size(actual_image, 1)
        for j=1:size(actual_image, 2)
            if ref_image(i, j) == 255 && actual_image(i, j) == 0
                foreground_error = foreground_error + 1;
            elseif ref_image(i, j) == 0 && actual_image(i, j) == 1
                background_error = background_error + 1;
            end
        end
    end
    
    error_foreground = (foreground_error / foreground_pixels) * p_fore;
    error_background = (background_error / background_pixels) * p_back;
    error = (error_foreground + error_background) * 100;
end

% Function to calculate scalar Gaussian
function prob_density = gaussian(x, u, sigma)
        prob_density = (1/sqrt(2*pi*sigma*sigma) * exp(-(x-u).^2/(2*sigma.^2)));
end

% Function for z-test
function distance=z_test(var1, mu1, var2, mu2)
        diff = mu1-mu2;
        denom =  sqrt(var1+var2);
        distance = abs(diff/denom); 
end
