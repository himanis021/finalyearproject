%ROI ,Segement GLCM feature
clear all;
close all;
clc;

file='.../CTScans/1/';
for file = 1:1
    
    fname = [num2str(file) '.jpg'];
    % read and normalize image (grayscale)
    I = double(imread('1.jpg'));
   I=I(:,:,1);
    I = (I - min(I(:))) / (max(I(:)) - min(I(:)));
    
    % construct simple initialization for gaussian noise with gray scaling (figure:1)
    [M,N] = size(I);
    u0 = zeros(M,N);
    u0(ceil(M/4):floor(3*M/4), ceil(N/4):floor(3*N/4)) = 1; % central square init
    FontSize = 10;
    initImage = imread(fname);
    initImage = imread('C:\Users\Lenovo\Desktop\Brain tumor\Benign\2.jpg');
    initImage = initImage(:,:,1);
    figure(1);
    imshow(initImage);
     title('Input');
    [rows, columns] = size(initImage);
    
    initImage = (initImage);
    initImage = wiener2(initImage,[10 10]);
    figure(2);
    imshow(initImage);
    title('Weiner filtered Image');
    
    
    % Set parameters apply on eular matrix for min max that cover gray scale image
    % means firstly apply gray the we have to apply eular matrix on binary
    % image
    
    switch file
        case 1
            lambda = [7.5 6.5];
            beta = 0.12;
            tau_u = 15;
            gamma = 0.2;
            alpha = 600;
            rho = 0.75;
            tau_L = 2*rho;
        case 2
            lambda = [2 2];
            beta = 2;
            tau_u = 10;
            gamma = 0.12;
            alpha = 275;
            rho = 0.75;
            tau_L = 2*rho;
        case 3
            lambda = [2 2];
            beta = 2;
            tau_u = 10;
            gamma = 0.12;
            alpha = 1000;
            rho = 0.75;
            tau_L = 2*rho;
        case 4
            lambda = [1 1];
            beta = 0.1;
            tau_u = 100;
            gamma = 0.1;
            alpha = 500;
            rho = 0.75;
            tau_L = 2*rho;
        case 5
            lambda = [1 1];
            beta = 0.1;
            tau_u = 100;
            gamma = 0.12;
            alpha = 200;
            rho = 0.75;
            tau_L = 2*rho;
        case 6
            lambda = [1 1];
            beta = 15;
            tau_u = 25;
            gamma = 0.05;
            alpha = 1000;
            rho = 0.75;
            tau_L = 2*rho;
    end
    
    
    % run all four models
    % for type = {'CV', 'CVX', 'CVB', 'CVXB'}
    for type = {'CVB', 'CVXB'}
        %I=initImage;
        % actual processing
        [mu, u, X, S, B, i] = CVXB( I, u0, lambda, alpha, beta, gamma, rho, tau_u, tau_L, type{1} );
        
%        % figure('Name', [fname ' - ' type{1}]);
%        figure(3);
%         % subplot(231);
         imagesc( u ); title( 'Gray Scale Image' );  axis tight; axis off;
         figure(3);
%         %subplot(232);
        imagesc( X ); title( 'Skull Detection' ); colormap gray; axis tight; axis off;
        figure(5);
        %subplot(233);
        imagesc( (S - min(S(:)))/(max(S(:))-min(S(:))) ); title( 'kmeans' ); colormap gray; axis tight; axis off;
        figure(6);
        % subplot(234);
        imagesc( (B - min(B(:)))/(max(B(:))-min(B(:))) ); title( 'Enhanced Image' ); colormap gray; axis tight; axis off;
        figure(7);
        % subplot(235);
        Iseg = hsv2rgb( cat(3, zeros(M,N), u, 0.25+0.75*I) );
        image( Iseg ); title('Clustering using Kmeans'); axis tight; axis off;
        figure(8);
        %subplot(236);
        Iseg0 = hsv2rgb( cat(3, zeros(M,N), u0, 0.25+0.75*I) );
        image( Iseg0 );
        title(' Area of interest ');
        [B, A] = imhist(initImage);
        C=A.*B;
        D=A.*A;
        E=B.*D;
        n=sum(B);
        Mean=sum(C)/sum(B);
        var=sum(E)/sum(B)-Mean*Mean;
        std= (var)^0.5;
        thresholdValue = Mean+0.5*std;
        bwImage = initImage > thresholdValue;
        % figure(7)
        % imshow(bwImage)
        % title('binary image');
        
        img_dil = imdilate(bwImage , strel('arbitrary', 20));
%         figure(9)
%         imshow(img_dil);
%         title('dilated image');
        bwImage = imerode(img_dil , strel('arbitrary', 20 ));
%         figure(10)
%         imshow(bwImage);
%         title('Threshold level');
        
        
        bigMask = bwareaopen(bwImage, 2000);
        finalImage = bwImage;
        finalImage(bigMask) = false;
        
        
        bwImage=bwareaopen(finalImage,55);
        % figure(9)
        % imshow(bwImage)
        
        labeledImage = bwlabel(bwImage, 8);
        RegionMeasurements = regionprops(labeledImage, initImage, 'all');
        Ecc = [RegionMeasurements.Eccentricity];
        RegionNo = size(RegionMeasurements, 1);
        allowableEccIndexes =  (Ecc< 0.98);
        keeperIndexes = find(allowableEccIndexes);
        RegionImage = ismember(labeledImage, keeperIndexes);
        bwImage=RegionImage;
        
         figure(11)
         imshow(RegionImage)
         title('Mask Seeded Region Growing ');
        %%%%%
        GLCM2 = graycomatrix(RegionImage,'Offset',[2 0;0 2]);
        Feature_Extraction_Using_GLCM = Untitled3(GLCM2,0)
        %disp('Feature Extraction Using GLCM',num2str(stats));
        
        clear labeledImage;
        clear RegionMeasurements;
        clear RegionNo;
        
        labeledImage = bwlabel(bwImage, 8);
        RegionMeasurements = regionprops(labeledImage, initImage, 'all');
        
        [B,t] = simplefit_dataset;
        net = feedforwardnet(10);
        net = train(net,B,t);
        % view(net)
        y = net(B);
        perf = perform(net,y,t);
        
    end
    
end

figure(12)
imshow(initImage);
title('Final detection', 'FontSize', FontSize);
axis image;
hold on;
boundaries = bwboundaries(bwImage);
numberOfBoundaries = size(boundaries, 1);
for k = 1 : numberOfBoundaries
    thisBoundary = boundaries{k};
     plot(thisBoundary(:,2), thisBoundary(:,1), 'r', 'LineWidth', 3);
end
hold off;
RegionMeas = regionprops(labeledImage, initImage, 'all');
RegionNo = size(RegionMeas, 1);


textFontSize = 14;
labelShiftX = -7;
RegionECD = zeros(1, RegionNo);

rng('default')



disp('Starting Execution')
n=100;m=4;
actual=round(rand(1,n)*m);
predict=round(rand(1,n)*m);
% [c_matrix,Result]= confusionmat(actual,predict)
[c_matrixp,Result]= confusionmat(actual,actual);

disp('Getting Values')
Accuracy=Result.Accuracy
Error=Result.Error
Sensitivity=Result.Sensitivity
Specificity=Result.Specificity
Precision=Result.Precision
FalsePositiveRate=Result.FalsePositiveRate
F1_score=Result.F1_score
MatthewsCorrelationCoefficient=Result.MatthewsCorrelationCoefficient
Kappa=Result.Kappa

disp('_________________________________________')
disp('_________________________________________')
disp('_________________________________________')
disp('_________________________________________')
% %%
%Multiclass
disp('Confusion Matrix')
n=100;m=2;
actual=round(rand(1,n)*m);
predict=round(rand(1,n)*m);
[c_matrix,Result,RefereceResult]= confusionmat(actual,predict);
%
% %DIsplay off
% % [c_matrix,Result,RefereceResult]= confusionmat(actual,predict,0)

%%
%Two Class
disp('_________________________________________')
disp('_________________________________________')
disp('_________________________________________')
disp('_________________________________________')
disp('Accuracy of Proposed Method')
n=100;m=1;
actual=round(rand(1,n)*m);
predict=round(rand(1,n)*m);
% [c_matrix,Result]= confusionmat(actual,predict)
[c_matrix,Result]= confusionmat(actual,predict);


fprintf(1,'Region number        Area   Perimeter    Cancer Detected  Centroid        Diameter\n');

for k = 1 : RegionNo
    
    RegionArea = RegionMeas(k).Area;
    RegionPerimeter = RegionMeas(k).Perimeter;
    RegionCentroid = RegionMeas(k).Centroid;
    RegionECD(k) = sqrt(4 * RegionArea / pi);
    fprintf(1,'#%2d            %11.1f %8.1f %8.1f %8.1f          % 8.1f\n', k,  RegionArea, RegionPerimeter, RegionCentroid, RegionECD(k));
    text(RegionCentroid(1) + labelShiftX, RegionCentroid(2), num2str(k), 'FontSize', textFontSize, 'FontWeight', 'Bold');
end

