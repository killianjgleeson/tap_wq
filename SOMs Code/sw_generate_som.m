clc
clear all
tic;

fid2=fopen('numer_attr_names.csv');
som_numer_attr_names=textscan(fid2,'%s','delimiter',',');
som_numer_attr_names=som_numer_attr_names{1}
som_numer_attrs=importdata('numer_attrs.csv');
fid3=fopen('categ_attr_names.csv');
som_categ_attr_names=textscan(fid3,'%s','delimiter',',');
som_categ_attr_names=som_categ_attr_names{1};
fid4=fopen('categ_attrs.csv');
categ_attrs_no=size(som_categ_attr_names,1);
som_categ_attrs=textscan(fid4,repmat('%s',1,categ_attrs_no),'delimiter',',');
fid5=fopen('numer_attr_units.csv');
som_numer_attr_units=textscan(fid5,'%s','delimiter',',');
som_numer_attr_units=som_numer_attr_units{1};
som_samples_id=som_categ_attrs{1};

for i=1:numel(som_numer_attr_names)
    lqu=0.05;
    uqu=0.95;
    lbounds(i)=quantile(som_numer_attrs(:,i), lqu);
    ubounds(i)=quantile(som_numer_attrs(:,i), uqu);
end
% lbounds(2)=0.05;
% ubounds(2)=0.28;
% lbounds(6)=1.0;
% ubounds(3)=30.0;
% lbounds(end)=0.0;
% ubounds(end)=8.0;

% [som_numer_attrs(:,2), outliers_idx] = outliers(som_numer_attrs(:,1), 10);
som_data=som_data_struct(som_numer_attrs,'comp_names',som_numer_attr_names,'labels',som_categ_attrs{1});
som_data=som_normalize(som_data,'range');

lbounds_norm=som_normalize(lbounds, som_data.comp_norm);
ubounds_norm=som_normalize(ubounds, som_data.comp_norm);
for i=1:length(lbounds_norm)
    if lbounds_norm(i)>=ubounds_norm(i)
        ubounds_norm(i)=lbounds_norm(i)+0.001;
    end
end

[som_map,P_sparse]=som_make(som_data,'imp');

figure('units', 'normalized', 'outerposition', [0 0 1 1]);
colormap(jet);
h_comp_plane_fig = som_show(som_map, ...
        'umat', 'all', ...
        'comp', 1:length(som_numer_attr_names), ...
        'norm', 'd', ...
        'footnote', '');

for i = 1:numel(som_numer_attr_names)
%   Label x axis of component planes with units
    ylabel(h_comp_plane_fig.plane(i + 1), ...
           som_numer_attr_units(i));
    fprintf('Limiting bounds of %s to [%.3f %.3f]\n', ...
        som_numer_attr_names{i}, lbounds(i), ubounds(i));
    set(h_comp_plane_fig.plane(i + 1), 'CLim', ...
        [lbounds_norm(i) ubounds_norm(i)]);
    set(h_comp_plane_fig.colorbar(i + 1),  ...
        'YTick', linspace(lbounds_norm(i), ubounds_norm(i), 4),  ...
        'YTickLabel', cellstr(num2str(linspace(lbounds(i), ubounds(i), 4)', '%10.2f\n')));
    set(get(h_comp_plane_fig.colorbar(i + 1),'label'),'string',som_numer_attr_units(i));

end

% Produce labelled SOMs for each categorical variable
label_attr_idx=1:numel(som_categ_attr_names);
for attr_idx = label_attr_idx
    attr_name = som_categ_attrs{attr_idx};
    attr_descr = som_categ_attr_names{attr_idx};

    % Label data (after replacing '?' entries with the null string '' to
    % ensure they are not considered in the cell labelling voting process)
%     labels = strrep(cellstr(tab.(attr_name)), '?', '');
    for k=1:size(attr_name,1)
        labels(k,1)=cellstr(attr_name{k});
        if strcmp(labels{k,1},'##'); labels{k,1}='#';end;
    end
    som_data = som_label(som_data, 'add', 'all', labels);

    % Autolabel map using label voting method
    som_map = som_autolabel(som_map, som_data, 'vote');

    fprintf('SOM labelled using %s\n', attr_descr);
    % New figure
    fig_h = figure('units', 'normalized', ...
                   'outerposition', [0 0, 0.6, 1]);
    som_show(som_map, 'empty', attr_descr, ...
             'subplots', [1, 1], ...
             'footnote', '');
    % Add labels to plot
    som_show_add('label', som_map, ...
                 'TextSize', 8, ...
                 'TextColor', 'black');
    % Colour each cell by unique label
    sw_labelled_som_color(gca(), som_map)  
    som_map = som_label(som_map, 'clear', 'all');
    som_data = som_label(som_data, 'clear', 'all');
end

%% Extraction of sample IDs from clustures
% The ID of those samples with a specific attribute 'less than', 'between'
% or 'greater than' a specific value (quntile of values of the attribute 
% at clusters/neurons) are extracted from the clusters.
% 
% ext_attr_name = 'Fe';
% arg = 'greater than';
% switch arg,
%     case 'less than',       qu_perc = 0.1;
%     case 'between',         qu_perc = [0.1 0.9];
%     case 'greater than',    qu_perc = 0.98;
% end
% 
% ext_attr_idx=find(strcmp(som_numer_attr_names,ext_attr_name));
% if ~isempty(ext_attr_idx)
%     if ~isempty(P_sparse) 
%         [neuron_samples,ext_attr_bound,ext_samples_num] = som_de_som(som_data,som_map,P_sparse,ext_attr_idx,arg,qu_perc);
%     else
%         fprintf('neuron_samples cannot be assigned since P_sparse is empty. See above for the reason." \n');
%     end
%     ext_samples_id=som_samples_id(ext_samples_num,:);
% 
%     fid_ext=fopen('ext_samples_id.csv','w+');
%     switch arg,
%         case 'less than',       fprintf(fid_ext,'ID of those samples associated with the neurons with %s lower than %g %s\n', ...
%                                 som_numer_attr_names{ext_attr_idx},ext_attr_bound, ...
%                                 som_numer_attr_units{ext_attr_idx});
%         case 'between',         fprintf(fid_ext,'ID of those samples associated with the neurons with %s between %g and %g %s\n', ...
%                                 som_numer_attr_names{ext_attr_idx},ext_attr_bound(1), ...
%                                 ext_attr_bound(2),som_numer_attr_units{ext_attr_idx});
%         case 'greater than',    fprintf(fid_ext,'ID of those samples associated with the neurons with %s higher than %g %s\n', ...
%                                 som_numer_attr_names{ext_attr_idx},ext_attr_bound, ...
%                                 som_numer_attr_units{ext_attr_idx}); 
%     end
%     for i=1:size(ext_samples_id,1)
%           fprintf(fid_ext,'%s\n',ext_samples_id{i});
%     end
%     fclose(fid_ext);
% end

toc