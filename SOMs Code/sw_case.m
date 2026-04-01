function [Case] = sw_case()

% Case = struct('filename',{''},...
%     'rawdata_dim',[],...
%     'colno_id',[],...
%     'colno_date',[],...
%     'colno_num_att',[],...
%     'colno_cat_att',[],...
%     'n_reps',[],...
%     'rep_colno',[],...
%     'rep_colno_n',[],...
%     'rep_orgs',cell(1),...
%     'rep_news',cell(1),...
%     'lab_filter_colno',[],...
%     'lab_filters',cell(1));

Case.filename = 'WSZ WQ samples dataset 2009-2022_includingnewCNT_rev3.csv';

Case.rawdata_dim = [177 169352]; 
Case.colno_id = 1;
Case.colno_date = [];
Case.date_bounds=[]; 
Case.colno_num_att = [12 17 13 20 118 169]; % 
Case.colno_cat_att = [34 67];

%%
i=0;
Case.n_reps = 0;

i = i+1;
Case.rep_colno(i) = 2;
categ = {'North','West','South','East'};
label = {'N','W','S','E'};
[Case] = sw_replace_categ_attrs (i,Case,categ,label);

i = i+1;
Case.rep_colno(i) = 3;
categ = {'Chlorine','Chloramination'};
label = {'N','M'};
[Case] = sw_replace_categ_attrs (i,Case,categ,label);

% if i~=Case.n_reps; error('Error in number of replacements'); end

%%
i=0;
Case.or_n_filts = 0;

i = i+1;
Case.or_filt_colno(i,1) = 4;
filter = {'all'}; % 
[Case] = sw_filter_or (i,Case,filter);

i = i+1;
Case.or_filt_colno(i,1) = 5;
filter = {'all'}; % 
[Case] = sw_filter_or (i,Case,filter);

i = i+1;
Case.or_filt_colno(i,1) = 6;
filter = {'all'}; % 
[Case] = sw_filter_or (i,Case,filter);

% if i~=Case.or_n_filts; error('Error in number of OR filters'); end

%%
i=0;
Case.and_n_filts = 0;

i = i+1;
Case.and_filt_colno(i,1) = 2;
filter = {'all'};
[Case] = sw_filter_and (i,Case,filter);

i = i+1;
Case.and_filt_colno(i,1) = 3;
filter = {'all'};
[Case] = sw_filter_and (i,Case,filter);

% if i~=Case.and_n_filts; error('Error in number of AND filters'); end
