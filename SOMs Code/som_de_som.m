function [neuron_samples,ext_attr_bound,ext_samples_num] = som_de_som(som_data,som_map,P_sparse,ext_attr_idx,arg,qu_perc)

samples_no=size(som_data.data,1);
neurons_no=size(som_map.codebook,1);
neuron_samples=zeros(neurons_no,1);
for i=1:neurons_no
    nd=0;
    for j=1:samples_no
        if (P_sparse(i,j)~=0)
            nd=nd+1;
            neuron_samples(i,nd)=j;
        end
    end
end
som_map_denorm = som_denormalize(som_map);
ext_attr_denorm=som_map_denorm.codebook(:,ext_attr_idx);
switch arg, 
    case 'less than',       ext_attr_bound=quantile(ext_attr_denorm, qu_perc); 
    case 'between',         ext_attr_bound=quantile(ext_attr_denorm, qu_perc);
    case 'greater than',    ext_attr_bound=quantile(ext_attr_denorm, qu_perc);
end
ext_attr_bound=100;
switch arg,
    case 'less than',       ext_range=find(ext_attr_denorm<ext_attr_bound);
    case 'between',         ext_range=find(ext_attr_denorm>ext_attr_bound(1) ...
                            & ext_attr_denorm<ext_attr_bound(2));
    case 'greater than',    ext_range=find(ext_attr_denorm>ext_attr_bound);
end

k=0;
for i=1:length(ext_range)
    for j=1:size(neuron_samples,2)
        id=neuron_samples(ext_range(i),j);
        if id~=0
            k=k+1;
            ext_samples_num(k)=id;
        end
    end
end
ext_samples_num=sort(ext_samples_num);
