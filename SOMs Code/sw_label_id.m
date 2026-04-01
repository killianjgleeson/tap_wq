function [categ_attrs] = sw_label_id(categ_attrs,attr_colno,attr_names,id)

for i=1:size(categ_attrs,1)
    for j=1:length(attr_names)
        kk=regexp(categ_attrs(i,attr_colno),attr_names(j));
        kl=length(kk{1});
        if kl==0; categ_attrs(i,attr_colno)=id(j); end
    end
end