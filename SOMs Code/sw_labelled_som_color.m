function sw_labelled_som_color(ax_h, som_map)
    % Get cell array of labels assigned to each output cell
    labs = som_map.labels;
    unique_labs = sort(unique(labs));
    
    % Create a mapping from each distinct label to a distinct RGB triple
    cmap = containers.Map(unique_labs, num2cell(hsv(length(unique_labs)), 2));
    % Empty array to store RGB face colours per output cell
    cdata = zeros(length(labs), 3);
    % Identify a face colour per cell using the above mapping
    for i = 1:length(labs)
        cdata(i, :) = cmap(char(labs(i)));    
    end
    
    % Change colour range from [0 1] to [0.5 1]
    cdata = (cdata + 1.0) / 2.0;
   
    % Set Face color data
    ax_h.Children(end).FaceVertexCData = cdata;
    % Enable flat shading per Face
    ax_h.Children(end).FaceColor = 'flat';
end