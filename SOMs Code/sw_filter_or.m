function [Case] = sw_filter_or (i,Case,filter)

Case.or_filters(i,1:length(filter)) = filter;
Case.or_filt_eles_n(i,1)=length(filter);

