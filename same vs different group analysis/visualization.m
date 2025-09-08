addpath('/Volumes/homes/Maya/Code/matlab');

data_path = '/Volumes/homes/Maya/Guests/Amit/CompSagolProj/correlation_output/normalized_data'; % directory for both loading and saving data
atlas_path = '/Volumes/homes/Maya/Atlases/schaefer_parcellation/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.dtseries.nii';

csv_files = dir(fullfile(data_path,'*.csv'));

for i = 1:length(csv_files)
    % generate the output file path
    output_filename = erase(csv_files(i).name, '.csv');
    output_path = char(string(data_path) + "/" + string(output_filename) + ".dtseries.nii");
    
    % load data
    data_filename = fullfile(data_path, csv_files(i).name);
    data = readmatrix(data_filename);
    
    data(:,1:1) = []; % remove first column of subject ids
    data(1:1,:) = []; % remove first row of region ids
    
    mean_vector = mean(data, 1);
    
    % create and save the cifti file
    surface_map(mean_vector, atlas_path, output_path);

end

