clc
clear all
tic;

%% Read Case
[Case] = sw_case();

%% Read datafile
data_table=readtable(Case.filename,'ReadVariableNames',false,'Format','auto');%,'Delimiter',',','ReadVariableNames',false,'Format','auto');
data_table(:,Case.rawdata_dim(1)+1:end)=[];
data_table(Case.rawdata_dim(2)+1:end,:)=[];

WSZ="CATTERICK/RICHMD 2019 WSZ";
Dis="Clm";
data_year=2009;

for i=1:length(Case.colno_num_att)
    numer_attr_names(i)=table2cell(data_table(2,Case.colno_num_att(i)));
    numer_attr(:,i)=(data_table(3:end,Case.colno_num_att(i)));
%     numer_attr=table2cell(data_table(3:end,Case.colno_num_att(i)));
%     numer_attrs(:,i)=str2double(numer_attr);
end


%% year selection--------------
% if Case.colno_num_att (2)==36; 
% date_time=datetime(numer_attr.(1));yr=year(date_time);
% numer_attr.(1)=yr;
% numer_attr=numer_attr((numer_attr.(1)==(data_year)),:);
% numer_attr=numer_attr(:,2:end);
% end %% year selection ----------------

%% WSZ selection--------------
% if Case.colno_num_att (3)==5; 
% numer_attr=numer_attr((numer_attr.(1)==(WSZ)),:);
% numer_attr=numer_attr(:,2:end);
% end %% WSZ selection ----------------

%% Disinfection selection--------------
% if Case.colno_num_att (1)==74; 
% numer_attr=numer_attr((numer_attr.(1)==(Dis)),:);
% numer_attr=numer_attr(:,2:end);
% end %% disinfection selection ----------------

numer_attr=table2array(numer_attr);
numer_attrs=str2double(numer_attr);
%% only iron samples
naniron=find(isnan(numer_attrs(:,3))); numer_attrs(naniron,:) = []; 
%% end %% only iron samples

%% only iron failures
% ironfails=numer_attrs(:,3)<100; numer_attrs(ironfails,:) = [];
%% end % only iron failures

%% only low edge btw
% btw=numer_attrs(:,6)>500; numer_attrs(btw,:) = [];
%% end % only low edge btw

% %% only high flow DMAs
% lowflow=numer_attrs(:,1)<100; numer_attrs(lowflow,:) = [];
% numer_attrs=numer_attrs(:,2:end);
% %% end % only high flow DMAs

%% For flushing data only %%
% Fl_Ev=206; %% F1=206 F2=207
% Days1=-365; Days2=365;
% 
% if Case.colno_num_att (end)==Fl_Ev
% a=numer_attrs(:,end)>=Days1 & numer_attrs(:,end)<=Days2;
% numer_attrs=numer_attrs(a,:);
% end
%% end fushing information

%% Units table
unit_tables=readtable('Sample Data 2009-2022 Variables_REV1.csv');
if Case.colno_num_att (1)==74; 
unit_tables=unit_tables(:,numer_attr_names(:,2:end));
elseif  Case.colno_num_att (1)==74 && Case.colno_num_att (2)==36 && Case.colno_num_att (3)==5;
unit_tables=unit_tables(:,numer_attr_names(:,3:end)); 
elseif  Case.colno_num_att (1)==74 && Case.colno_num_att (2)==3 && Case.colno_num_att (3)==70;
unit_tables=unit_tables(:,numer_attr_names(:,4:end));
elseif  Case.colno_num_att (1)==135;
unit_tables=unit_tables(:,numer_attr_names(:,2:end));
else
unit_tables=unit_tables(:,numer_attr_names); 
end
numer_attr_names=unit_tables(1,:); numer_attr_names=table2cell(numer_attr_names);
numer_attr_units=unit_tables(2,:); numer_attr_units=table2cell(numer_attr_units);
% numer_attr_units={'mg/l','µg/l','µg/l',...
%     'pH','C','NTU','mm','years'};

for i=1:length(Case.colno_cat_att)
    categ_attr_names(i)=table2cell(data_table(2,Case.colno_cat_att(i)));
    categ_attrs(:,i)=(data_table(3:end,Case.colno_cat_att(i)));
end
% if Case.colno_cat_att (1)==74;
% categ_attrs=categ_attrs((categ_attrs.(1)==(Dis)),:);
% end
if Case.colno_num_att (2)==36;
categ_attrs=categ_attrs((categ_attrs.(4)==(data_year)),:);
end
% if Case.colno_cat_att (3)==64;
% categ_attrs(naniron,:) = []; 
% end

%% --------NAN iron and flushing
% categ_attrs(~naniron,:)=[];   %%iron
% categ_attrs(~a,:)=[];     %flushing
%% --------end NAN iron and flushing
categ_attrs=table2cell(categ_attrs);
categ_attrs(naniron,:) = [];
categ_attrs=strrep(categ_attrs,',','-');
empties=cellfun('isempty',categ_attrs);
categ_attrs(empties)={'#'};

sample_ids=table2cell(data_table(3:end,Case.colno_id));

dates=table2cell(data_table(3:end,Case.colno_date));
sample_date=datetime(dates,'InputFormat','dd-MMM-yyyy')+years(2000);

%%
for k1=1:Case.n_reps
    j=Case.rep_colno(k1);
    for k2=1:Case.rep_orgs_n(k1,1)
        k_org=Case.rep_orgs(k1,k2);
        k_new=Case.rep_news(k1,k2);
        categ_attrs(:,j)=strrep(categ_attrs(:,j),k_org,k_new);
    end
end

%%
%% remove when all categ row is nan
samples_rem=zeros(size(categ_attrs,1),1);
fulls=(~isnan(numer_attrs));
nan_rows=find(sum(fulls,2)==0); 
if ~isempty(nan_rows)
    samples_rem(nan_rows,1)=1;
end
rem_vec=find(samples_rem==1);
numer_attrs(rem_vec,:)=[];
categ_attrs(rem_vec,:)=[];
if ~isempty(sample_date); sample_date(rem_vec,:)=[]; end
sample_ids(rem_vec,:)=[];

%% remove dates
samples_rem=zeros(size(categ_attrs,1),1);
if ~isempty(sample_date) && ~isempty(Case.date_bounds)
    rem_dates=find(year(sample_date)<Case.date_bounds(1) | year(sample_date)>Case.date_bounds(2));
    if ~isempty(rem_dates)
        samples_rem(rem_dates,1)=1;
    end
end
rem_vec=find(sum(samples_rem,2)==1);
numer_attrs(rem_vec,:)=[];
categ_attrs(rem_vec,:)=[];
if ~isempty(sample_date); sample_date(rem_vec,:)=[]; end
sample_ids(rem_vec,:)=[];

%% remove or_filters
if Case.or_n_filts~=0
    samples_sav=zeros(size(categ_attrs,1),1);
    for k1=1:Case.or_n_filts
        if not(strcmp(Case.or_filters(k1),'all'))
            j=Case.or_filt_colno(k1);
            samples_sav(:,k1)=ismember(categ_attrs(:,j),Case.or_filters(k1,1:Case.or_filt_eles_n(k1)));
        end
    end
    rem_vec=find(sum(samples_sav,2)==0);
    numer_attrs(rem_vec,:)=[];
    categ_attrs(rem_vec,:)=[];
    if ~isempty(sample_date); sample_date(rem_vec,:)=[]; end
    sample_ids(rem_vec,:)=[];
end

%% remove and_filters
if Case.and_n_filts~=0
    samples_rem=zeros(size(categ_attrs,1),1);
    for k1=1:Case.and_n_filts
        if not(strcmp(Case.and_filters(k1),'all'))
            j=Case.and_filt_colno(k1);
            samples_rem(:,k1)=~ismember(categ_attrs(:,j),Case.and_filters(k1,1:Case.or_filt_eles_n(k1)));
        end
    end
    rem_vec=find(sum(samples_rem,2)~=0);
    numer_attrs(rem_vec,:)=[];
    categ_attrs(rem_vec,:)=[];
    if ~isempty(sample_date); sample_date(rem_vec,:)=[]; end
    sample_ids(rem_vec,:)=[];
end

%%
% end_categ_attrs=size(categ_attrs,2);
% WTW_no=3;
% WTW_cols=[4 5 6];
% for i=1:size(categ_attrs,1)
%     aa=categ_attrs{i,WTW_cols(1)}(1);
%     bb=categ_attrs{i,WTW_cols(2)}(1);
%     cc=categ_attrs{i,WTW_cols(3)}(1);
% %     cc=[categ_attrs{i,WTW_cols(3)}(1) categ_attrs{i,WTW_cols(3)}(end)];
%     if strcmp(aa,'#'); aa='';end;
%     if strcmp(bb,'#'); bb='';end;
%     if strcmp(cc,'#'); cc='';end;
%     categ_attrs{i,end_categ_attrs+1}=[aa bb cc];
% end
% categ_attr_names{1,end_categ_attrs+1}='WTW_comb';

%%
% multiplying cast by 100 to get the percentage
percent_attrs=[cellstr('pc CI - Cast Fe'), cellstr('pc DI - Ductile Fe'), cellstr('pc SI - Spun Fe')];
for i=1:numel(numer_attr_names)
    for j=1:numel(percent_attrs)
        if strcmp(numer_attr_names(i),percent_attrs(j))
            numer_attrs(:,i)=numer_attrs(:,i)*100;
            break
        end
    end
end

%%
% % combining cast + spun iron
% kk=0;
% for i=1:numel(numer_attr_names)
%     if strcmp(numer_attr_names(i),percent_attrs(1)) ...
%             || strcmp(numer_attr_names(i),percent_attrs(3))
%         kk=kk+1;
%         idx_cast(kk)=i;
%     end
% end
% if length(idx_cast)==2
%     numer_attrs(:,idx_cast(1))= ...
%         numer_attrs(:,idx_cast(1))+numer_attrs(:,idx_cast(2));
%     numer_attr_names(idx_cast(1))=cellstr('pc CI Cast Fe + Spun Fe');
%     numer_attrs(:,idx_cast(2:end))=[];
%     numer_attr_names(idx_cast(2:end))=[];
% end

%%
dateformat = 'mm-dd-yy';
sample_date_out=cellstr(datestr(sample_date,dateformat));
categ_attrs(:,end+1)=sample_date_out;
categ_attr_names{1,end+1}='Sample date';

indata.ids=sample_ids;
indata.dates=sample_date_out;
indata.numer_attr_names=numer_attr_names;
indata.categ_attr_names=categ_attr_names;
indata.numer_attrs=numer_attrs;
indata.categ_attrs=categ_attrs;
indata.numer_attr_units=numer_attr_units;

fid1=fopen('numer_attr_names.csv','w+');
fid2=fopen('categ_attr_names.csv','w+');
fid3=fopen('numer_attrs.csv','w+');
fid4=fopen('categ_attrs.csv','w+');
fid5=fopen('numer_attr_units.csv','w+');
fprintf(fid1,'%s,',numer_attr_names{:});
fprintf(fid2, '%s,',categ_attr_names{:});
for i=1:size(numer_attrs,1)
      fprintf(fid3,'%f,',numer_attrs(i,1:end-1));
      fprintf(fid3,'%f\n',numer_attrs(i,end));
      fprintf(fid4,'%s,',categ_attrs{i,1:end-1});
      fprintf(fid4,'%s\n',categ_attrs{i,end});
end
fprintf(fid5,'%s,',numer_attr_units{:});
fclose(fid1);
fclose(fid2);
fclose(fid3);
fclose(fid4);
fclose(fid5);

toc
