
% Load training set 

TrainingSet = load("TrainingSamplesDCT_8.mat");
foreground = TrainingSet.TrainsampleDCT_FG;
background = TrainingSet.TrainsampleDCT_BG;

% ---------------------------------------------------------------------- %

% Problem a
foreground_samples = size(foreground, 1); % Calculating the number of foreground sampples
background_samples = size(background, 1); % Calculating the number of background samples

prior_foreground = foreground_samples/(foreground_samples+background_samples);
prior_background = background_samples/(foreground_samples+background_samples);

% ---------------------------------------------------------------------- %

% Problem b
binwidth = 1;
% Foreground histogram
foreground_index = zeros(1, foreground_samples);
for idx = 1:foreground_samples
    [fore_max, fore_index] = sort(abs(foreground(idx,:)), 'descend');
    foreground_index(idx) = fore_index(2);
end
figure;
histogram_foreground = histogram(foreground_index, 'BinWidth', binwidth);
histogram_foreground.Normalization = 'probability';
title("Histogram of foreground")
%histogram_fore_values = histogram_foreground.Values;
histogram_fore_values = zeros(1, 64/binwidth);
for i=1:size(histogram_foreground.Values,2)
    histogram_fore_values(i) = histogram_foreground.Values(i);
end

% Background histogram
background_index = zeros(1, background_samples);
for idx = 1:background_samples
    [max, index] = sort(abs(background(idx,:)), 'descend');
    background_index(idx) = index(2);
end
figure;
histogram_background = histogram(background_index);
histogram_background.Normalization = 'probability';
title("Histogram of background")
%histogram_back_values = histogram_background.Values;
histogram_back_values = zeros(1, 64/binwidth);
for i=1:size(histogram_background.Values,2)
    histogram_back_values(i) = histogram_background.Values(i);
end

% ---------------------------------------------------------------------- %
% Problem c
pad_value = 7;
cheetah = imread("cheetah.bmp");
cheetah = padarray(cheetah,[pad_value pad_value], 'post');
cheetah = im2double(cheetah);

[rows, cols] = size(cheetah);

unpadded_rows = rows-pad_value;
unpadded_cols = cols - pad_value;
converted_block = zeros(unpadded_rows, unpadded_cols);
zigzag = load("Zig-Zag Pattern.txt");
zigzag = zigzag + 1; %Following MATLAB index
for i = 1:unpadded_rows
    for j = 1:unpadded_cols
        block = cheetah(i:i+7, j:j+7);
        transform = dct2(block); %Utilize inbuilt MATLAB function for DCT calculation
        % Create zigzag pattern
        zigzag_transform(zigzag) = transform;
        [value,index]=sort(abs(zigzag_transform),'descend'); %Sorting and taking the second index
        converted_block(i,j)=ceil(index(2)/binwidth);  % Sorting and taking the second index            
    end
end

% Create the mask
predicted_image = zeros(unpadded_rows, unpadded_cols);
for i = 1:unpadded_rows
    for j = 1:unpadded_cols
        cheetah_prob = histogram_fore_values(1, converted_block(i,j)) * prior_foreground;
        grass_prob = histogram_back_values(1, converted_block(i,j)) * prior_background;
        if (cheetah_prob >= grass_prob) 
            predicted_image(i,j) = 1;
        else
            predicted_image(i,j) = 0;
        end
    end
end
figure;
imagesc(predicted_image);
title('Prediction')
colormap(gray(255));
% --------------------------------------------------------------------- %
% Problem d

true_image = imread('cheetah_mask.bmp');
% Count the number of 1s and 0s
foreground_pixels = 0;
background_pixels = 0;

for i=1:size(true_image, 1)
    for j=1:size(true_image, 2)
        if true_image(i, j) == 255
            foreground_pixels = foreground_pixels + 1;
        else
            background_pixels = background_pixels + 1;
        end
    end
end

foreground_error = 0;
background_error = 0;
for i=1:size(predicted_image, 1)
    for j=1:size(predicted_image, 2)
        if true_image(i, j) == 255 && predicted_image(i, j) == 0
            foreground_error = foreground_error + 1;
        elseif true_image(i, j) == 0 && predicted_image(i, j) == 1
            background_error = background_error + 1;
        end
    end
end

error_foreground = (foreground_error / foreground_pixels) * prior_foreground;
error_background = (background_error / background_pixels) * prior_background;
total_error = (error_foreground + error_background) * 100;

X = ['Probability of error is: ', num2str(total_error), '%'];
disp(X)

