function sample_viewpoints(cls, azimuths, elevations, tilts, distances, ...
    num_samples, outlier_ratio, outlier_elevation_avg, outlier_elevation_std, outlier_tilt_avg, outlier_tilt_std)
    
% remove invalid samples
invalidSamples = find(distances == 0);
azimuths(invalidSamples) = [];
elevations(invalidSamples) = [];
tilts(invalidSamples) = [];
distances(invalidSamples) = [];

 % calibrate distance by a factor of 3, because estimated distances
 % from pascal3d annotations are over-estimated.
targetSamples = [azimuths', elevations', tilts', distances'/3];
bandwidth = 1.06 * std(targetSamples) * (size(targetSamples, 1)^(-0.2));

addpath('matlab_kde_package');
p = kde(targetSamples', bandwidth');
newSamples = sample(p, num_samples);
rmpath('matlab_kde_package');

% normalize viewpoints
goodSamples = newSamples';
goodSamples(:, 1) = mod(goodSamples(:, 1)+180, 360)-180;
goodSamples(:, 2) = mod(goodSamples(:, 2)+90, 180)-90;
goodSamples(:, 3) = mod(goodSamples(:, 3)+90, 180)-90;

% normalize distances
distance_min = min(distances);
distance_max = max(distances);
filt = goodSamples(:, 4) > distance_max | goodSamples(:, 4) < distance_min;
goodSamples(filt, 4) = datasample(goodSamples(~filt,4), sum(filt)) + 1*rand(sum(filt),1);

% add some outliers
rp = randperm(num_samples);
numRandPerturbation = int32(num_samples*outlier_ratio);
rp = rp(1:numRandPerturbation);

randAzimuth = rand(numRandPerturbation, 1) * 360 - 180;
randElevation = max(min(normrnd(outlier_elevation_avg, outlier_elevation_std, numRandPerturbation, 1), 85),-85);
randTilt = max(min(normrnd(outlier_tilt_avg, outlier_tilt_std, numRandPerturbation, 1), 45),-45);

goodSamples(rp, 1) = randAzimuth;
goodSamples(rp, 2) = randElevation;
goodSamples(rp, 3) = randTilt;

figure(1);
subplot(2,2,1), histogram(goodSamples(:,1), 32, 'Normalization', 'probability'), title([cls ' azimuth']);
subplot(2,2,2), histogram(goodSamples(:,2), 32,'Normalization', 'probability'), title([cls ' elevation']);
subplot(2,2,3), histogram(goodSamples(:,3), 32,'Normalization', 'probability'), title([cls ' tilt']);
subplot(2,2,4), histogram(goodSamples(:,4), 32,'Normalization', 'probability'), title([cls ' distance']);
pause(0.5);

outdir = 'view_distributions';
if exist(outdir, 'dir') == 0
    mkdir(outdir);
end
dlmwrite(fullfile(outdir, sprintf('%s.txt', cls)), goodSamples, 'delimiter', ' ', 'precision', 6);