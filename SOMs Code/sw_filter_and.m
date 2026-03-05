function [Case] = sw_filter_and (i,Case,filter)

Case.and_filters(i,1:length(filter)) = filter;
Case.and_filt_eles_n(i,1)=length(filter);
