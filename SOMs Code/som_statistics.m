%% SOMs statistics - Calculate number of samples per parameter, 
%% average value and standard deviatiom
clc;
tic;

attr0=["Samples" "Average" "Standard Deviation"];
attr1=sum(~isnan(som_numer_attrs),1);
attr2=nanmean(som_numer_attrs);
attr3=nanstd(som_numer_attrs);
attr4=[attr1;attr2;attr3];
attr=rand(3,numel(som_numer_attr_names)+1);
attr(1:3,2:numel(som_numer_attr_names)+1)=attr4;
attr=num2cell(attr);
attr{1,1}=attr0 (1); attr{2,1}=attr0 (2);attr{3,1}=attr0 (3);
attr_names=som_numer_attr_names'; attr_names=[repmat({'AA'},1) attr_names];
attr=cell2table(attr,'Variablenames',attr_names);
writetable(attr,"01_bact failures.xlsx");
toc;
