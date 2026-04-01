function [Case] = sw_replace_categ_attrs (i,Case,categ,label)

Case.rep_orgs(i,1:length(categ)) = categ;
Case.rep_news(i,1:length(categ)) = label;
Case.rep_orgs_n(i,1)=length(label);
