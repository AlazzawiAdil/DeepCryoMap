close all;
clear all;

% Binary_mask_images='C:\Users\Adil Al-Azzawi\Desktop\Particle Alignement\Particles Images\Apoferritin\Binary Masks';
Particles_images='C:\Users\Adil Al-Azzawi\Desktop\Particle Alignement\Alignement Results\KLH\Side View\Localized Aligning\Original\Particle Images';
% Particles_images1='C:\Users\Adil Al-Azzawi\Desktop\Particle Alignement\Particles Images\Original\Top_view';


% load Ribosome_Map.mat;

cd(Particles_images);
D1 = dir('*.png');
fixed  = ((imread(D1(1).name)));
% figure;imshow(I1);

load upToScaleReconstructionCameraParameters.mat

% Change the direction to the image folder
   for i = 2:numel(D1)
       tic;
        close all;
        % start to read cancer image
        clc; 
        disp('======================================================================================');        
        disp('  P  A  R  T  I  C  L  E -  D  E  T  E  C  T  I  O  N  - A N D -  P  I  C  K  I  N  G ');
        disp('  ------------- T R A I N I N G - D A T A S E T - P R E P E R A T I O N ------------- ');
        disp('======================================================================================');
        disp(' ');
        fprintf('Cryo-EM Image No. : %d\n',i');  
        
        %% Read the Particles Masks
        particle_image = (imread(D1(i).name));
        moving=particle_image;
        
        %% Image Registration 
        [optimizer,metric] = imregconfig('multimodal');
        tformSimilarity = imregtform(moving,fixed,'similarity',optimizer,metric);
        Rfixed = imref2d(size(fixed));
        movingRegisteredRigid = imwarp(moving,tformSimilarity,'OutputView',Rfixed);
        figure; imshowpair(movingRegisteredRigid, fixed); title('D: Based on Similarity Transformation Model')
        
        I2(:,:,1)=im2double(movingRegisteredRigid);
        I2(:,:,2)=im2double(movingRegisteredRigid);
        I2(:,:,3)=im2double(movingRegisteredRigid);
        figure; imshow(I2);
        
%         [m,n]=size(fixed);
%         I1=zeros(m,n,3);
        I1(:,:,1)=im2double(fixed);
        I1(:,:,2)=im2double(fixed);
        I1(:,:,3)=im2double(fixed);
        figure; imshowpair(I1,I2,'montage'); title('First Particle Frame and Second Registred Particle Frame ')

C = imfuse(fixed,moving,'falsecolor','Scaling','joint','ColorChannels',[1 2 0]);
figure; imshow(C);

C = imfuse(fixed,moving,'blend','Scaling','joint');
figure; imshow(C)
        %% Find Point Correspondences Between The Images
        % Detect feature points
        imagePoints1 = detectMinEigenFeatures((fixed), 'MinQuality', 0.1);

        % Visualize detected points
        figure
        imshow(I1, 'InitialMagnification', 50);
        title('Strongest Corners from the First Image');
        hold on
        plot((imagePoints1));

        % Create the point tracker
        tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);

        % Initialize the point tracker
        imagePoints1 = imagePoints1.Location;
        initialize(tracker, imagePoints1, I1);

        % Track the points
        [imagePoints2, validIdx] = step(tracker, I2);
        matchedPoints1 = imagePoints1(validIdx, :);
        matchedPoints2 = imagePoints2(validIdx, :);

        % Visualize correspondences
        figure
        showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2);
        title('Tracked Features');
        
        %% Estimate the fundamental matrix
        [E, epipolarInliers] = estimateEssentialMatrix(...
            matchedPoints1, matchedPoints2, cameraParams, 'Confidence', 99.99);

        % Find epipolar inliers
        inlierPoints1 = matchedPoints1(epipolarInliers, :);
        inlierPoints2 = matchedPoints2(epipolarInliers, :);

        % Display inlier matches
        figure
        showMatchedFeatures(I1, I2, inlierPoints1, inlierPoints2);
        title('Epipolar Inliers Matches');
        
        %% Compute the Camera Pose
        [orient, loc] = relativeCameraPose(E, cameraParams, inlierPoints1, inlierPoints2);

        %% Reconstruct the 3-D Locations of Matched Points
        % Detect dense feature points. Use an ROI to exclude points close to the
        % image edges.
        roi = [30, 30, size(I1, 2) - 30, size(I1, 1) - 30];
        imagePoints1 = detectMinEigenFeatures(rgb2gray(I1), 'ROI', roi, ...
            'MinQuality', 0.001);

        % Create the point tracker
        tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);

        % Initialize the point tracker
        imagePoints1 = imagePoints1.Location;
        initialize(tracker, imagePoints1, I1);

        % Track the points
        [imagePoints2, validIdx] = step(tracker, I2);
        matchedPoints1 = imagePoints1(validIdx, :);
        matchedPoints2 = imagePoints2(validIdx, :);

        % Compute the camera matrices for each position of the camera
        % The first camera is at the origin looking along the Z-axis. Thus, its
        % rotation matrix is identity, and its translation vector is 0.
        camMatrix1 = cameraMatrix(cameraParams, eye(3), [0 0 0]);

        % Compute extrinsics of the second camera
        [R, t] = cameraPoseToExtrinsics(orient, loc);
        camMatrix2 = cameraMatrix(cameraParams, R, t);

        % Compute the 3-D points
        points3D = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);

        % Get the color of each reconstructed point
        numPixels = size(I1, 1) * size(I1, 2);
        allColors = reshape(I1, [numPixels, 3]);
        colorIdx = sub2ind([size(I1, 1), size(I1, 2)], round(matchedPoints1(:,2)), ...
            round(matchedPoints1(:, 1)));
        color = allColors(colorIdx, :);

        % Create the point cloud
        ptCloud = pointCloud(points3D, 'Color', color);
        
        %% Display the 3-D Point Cloud
        % Visualize the camera locations and orientations
        cameraSize = 0.1;
        figure
        
%         plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
%         hold on
%         grid on
%         plotCamera('Location', loc, 'Orientation', orient, 'Size', cameraSize, ...
%             'Color', 'b', 'Label', '2', 'Opacity', 0);

        % Visualize the point cloud
        pcshow(ptCloud);
        
        
        
        
        pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
            'MarkerSize', 50);

        % Rotate and zoom the plot
        camorbit(0, 0);
        camzoom(1.5);

        % Label the axes
        xlabel('x-axis');
        ylabel('y-axis');
        zlabel('z-axis')
        title('3D Map Reconstruction ');
        
     pause;  
     fixed=moving;
   end
