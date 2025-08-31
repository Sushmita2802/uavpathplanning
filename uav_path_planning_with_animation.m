function uav_path_planning_with_animation()
    clc; close all; clearvars;

    %Map and obstacles
    mapSize = [50,50,20];           
    map = false(mapSize);

    % Buildings 
    map(15:25, 15:25, 1:12) = true;   % building 1
    map(28:36, 20:30, 1:16) = true;   % building 2
    map(20:30, 34:42, 1:10) = true;   % building 3

    inflation_radius = 0.5;  %acts as a safety margin
    map_infl = inflateMap(map, inflation_radius);

    %Initial and Final point
    start = [5,5,10];
    goal  = [45,45,10];

    % ensure start/goal free in inflated map
    if map_infl(start(1),start(2),start(3)) || map_infl(goal(1),goal(2),goal(3))
        error('Start or goal lies inside inflated obstacle. Choose different coords or reduce inflation.');
    end

    %A*(26)
    altPenalty = 6;     % vertical move penalty to prefer sideways avoidance
    [path3d, success, planTime] = astar3d_grid(map_infl, start, goal, altPenalty);
    if ~success
        error('A* failed to find a path.');
    end
    fprintf('A* found path: %d nodes, time %.3fs\n', size(path3d,1), planTime);

    %Shortcut smoothing
    path_short = shortcutPath(path3d, map_infl);

    %Spline smoothing (to get collision-free spline) 
       samples_list = [500, 400, 300, 200, 150, 120, 100, 80, 60, 40];
    safe_curve = [];
    for s = samples_list
        curve_try = splineSmooth(path_short, s);
        if ~curveCollides(curve_try, map_infl)
            safe_curve = curve_try;
            fprintf('Safe spline found with %d samples\n', s);
            break;
        end
    end
    if isempty(safe_curve)
        % fallback to the shortcut path (safe by construction)
        warning('Could not find collision-free spline. Falling back to shortcut path.');
        safe_curve = path_short;
    end

    %Visualization 
    fig = figure('Name','UAV Path Planning (safe)','Color','w');
    ax = axes(fig);
    hold(ax,'on'); grid(ax,'on'); view(3);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    xlim([1 mapSize(1)]); ylim([1 mapSize(2)]); zlim([1 mapSize(3)]);
    daspect([1 1 1]);
    rotate3d on;

    % raw A* path (red dashed)
    plot3(ax, path3d(:,1), path3d(:,2), path3d(:,3), 'r--', 'LineWidth', 1);
    % shortcut path (blue)
    plot3(ax, path_short(:,1), path_short(:,2), path_short(:,3), 'b-', 'LineWidth', 1.5);
    % final smooth curve path (green)
    plot3(ax, safe_curve(:,1), safe_curve(:,2), safe_curve(:,3), 'g-', 'LineWidth', 2.5);

    %for making buildings
    drawBuildings(map, 0.6);  % buildings, solid & semi-transparent
    drawBuildings(map_infl, 0.15);% inflated version, faint

    % start and goal markers
    plot3(ax, start(1),start(2),start(3), 'co', 'MarkerSize',10, 'MarkerFaceColor','c');
    plot3(ax, goal(1),goal(2),goal(3), 'mo', 'MarkerSize',10, 'MarkerFaceColor','m');

    title(sprintf('A*: %d -> shortcut: %d -> curve: %d', size(path3d,1), size(path_short,1), size(safe_curve,1)));
    drawnow;

    %Animation (quadrotor cross + camera follow)
    arm_len = 2.0;
    uav_arms = [-arm_len 0 0; arm_len 0 0; 0 -arm_len 0; 0 arm_len 0];
    hArms = plot3(ax, NaN, NaN, NaN, 'r-', 'LineWidth', 2);
    hCenter = plot3(ax, NaN, NaN, NaN, 'ro', 'MarkerFaceColor','r', 'MarkerSize',6);

    for k = 1:size(safe_curve,1)
        pos = safe_curve(k,:);
        body_pts = uav_arms + pos;
        set(hArms, 'XData', body_pts(:,1), 'YData', body_pts(:,2), 'ZData', body_pts(:,3));
        set(hCenter, 'XData', pos(1), 'YData', pos(2), 'ZData', pos(3));

        % camera follows behind and slightly above UAV
        campos(ax, pos + [12 12 8]);
        camtarget(ax, pos);
        camup(ax, [0 0 1]);

        drawnow;
        pause(0.03);
    end

    hold(ax,'off');
end

% inflateMap
function map_out = inflateMap(map_in, radius)
    % Inflate boolean voxel map by given radius (Euclidean-ish) using convn.
    % radius: integer >=0
    if radius <= 0
        map_out = map_in;
        return;
    end
    % kernel size
    k = 2*radius + 1;
    kernel = ones(k,k,k);
    convResult = convn(double(map_in), kernel, 'same');
    map_out = convResult > 0;
end

%3D A* (26)
function [path, success, t] = astar3d_grid(map, start, goal, altPenalty)
    if nargin < 4, altPenalty = 1; end
    tic;
    mapSize = size(map);
    N = prod(mapSize);

    startInd = sub2ind(mapSize, start(1), start(2), start(3));
    goalInd  = sub2ind(mapSize, goal(1),  goal(2),  goal(3));

    % generate 26 neighbor offsets
    [dx,dy,dz] = ndgrid(-1:1,-1:1,-1:1);
    moves = [dx(:), dy(:), dz(:)];
    moves(all(moves==0,2),:) = []; % remove 0,0,0

    g = inf(N,1);
    f = inf(N,1);
    parent = zeros(N,1,'int32');
    closed = false(N,1);

    g(startInd) = 0;
    f(startInd) = heuristic_euclid(start, goal);

    openInds = startInd;
    openF = f(startInd);

    maxIter = 5e6; iter = 0;
    while ~isempty(openInds)
        iter = iter + 1;
        if mod(iter,1e6)==0
            warning('A* still running... iter=%d', iter);
        end
        % pop lowest f
        [~, idx] = min(openF);
        curInd = openInds(idx);
        openInds(idx) = []; openF(idx) = [];

        if curInd == goalInd
            break;
        end

        closed(curInd) = true;
        [cr,cc,cz] = ind2sub(mapSize, curInd);

        for m = 1:size(moves,1)
            nr = cr + moves(m,1);
            nc = cc + moves(m,2);
            nz = cz + moves(m,3);
            if nr < 1 || nr > mapSize(1) || nc < 1 || nc > mapSize(2) || nz < 1 || nz > mapSize(3)
                continue;
            end
            nInd = sub2ind(mapSize, nr, nc, nz);
            if map(nInd), continue; end
            if closed(nInd), continue; end

            %penalize vertical moves
            dz = abs(nz - cz);
            moveCost = norm([moves(m,1), moves(m,2), moves(m,3)]);
            if dz > 0
                moveCost = moveCost * altPenalty;
            end

            tentative_g = g(curInd) + moveCost;

            if tentative_g < g(nInd)
                g(nInd) = tentative_g;
                parent(nInd) = int32(curInd);
                h = heuristic_euclid([nr nc nz], goal);
                f(nInd) = tentative_g + h;

                pos = find(openInds == nInd, 1);
                if isempty(pos)
                    openInds(end+1) = nInd; 
                    openF(end+1) = f(nInd); 
                else
                    openF(pos) = f(nInd);
                end
            end
        end
    end

    % reconstruct
    if parent(goalInd) == 0 && startInd ~= goalInd
        path = []; success = false; t = toc; return;
    end

    cur = goalInd;
    rev = zeros(0,1,'int32');
    while cur ~= 0
        rev(end+1) = cur; 
        cur = parent(cur);
    end
    rev = rev(end:-1:1);
    path = zeros(length(rev),3);
    for i=1:length(rev)
        [r,c,z] = ind2sub(mapSize, rev(i));
        path(i,:) = [r,c,z];
    end
    success = true;
    t = toc;
end

%heuristic (euclidean)
function h = heuristic_euclid(p, q)
    h = norm(double(p)-double(q));
end

%Shortcut smoothing (line-of-sight)
function shortPath = shortcutPath(path, map)
    if isempty(path), shortPath = path; return; end
    shortPath = path(1,:);
    i = 1;
    while i < size(path,1)
        j = size(path,1);
        while j > i+1
            if lineFree(path(i,:), path(j,:), map)
                break;
            end
            j = j - 1;
        end
        shortPath = [shortPath; path(j,:)]; 
        i = j;
    end
end

function ok = lineFree(p1,p2,map)
    % sample along straight segment and check voxels
    num = max(ceil(max(abs(p2-p1))*2), 2);
    xs = round(linspace(p1(1), p2(1), num));
    ys = round(linspace(p1(2), p2(2), num));
    zs = round(linspace(p1(3), p2(3), num));
    sz = size(map);
    ok = true;
    for k = 1:num
        x = max(1,min(sz(1), xs(k)));
        y = max(1,min(sz(2), ys(k)));
        z = max(1,min(sz(3), zs(k)));
        if map(x,y,z)
            ok = false; return;
        end
    end
end

%Spline smoothing (with safety check)
function curve = splineSmooth(path, samples)
    if nargin < 2, samples = 200; end
    if size(path,1) < 3
        curve = path; return;
    end
    t = [0; cumsum(sqrt(sum(diff(path).^2,2)))];
    ts = linspace(0, t(end), samples);
    curve(:,1) = spline(t, path(:,1), ts);
    curve(:,2) = spline(t, path(:,2), ts);
    curve(:,3) = spline(t, path(:,3), ts);
end

%check if spline curve collides with inflated map 
function coll = curveCollides(curve, map)
    % check samples and line segments
    coll = false;
    if isempty(curve), coll = true; return; end
    sz = size(map);
    % check sample points
    for i=1:size(curve,1)
        xi = round(max(1,min(sz(1),curve(i,1))));
        yi = round(max(1,min(sz(2),curve(i,2))));
        zi = round(max(1,min(sz(3),curve(i,3))));
        if map(xi,yi,zi)
            coll = true; return;
        end
    end
    % check line segments between consecutive samples
    for i=1:size(curve,1)-1
        if ~lineFree(round(curve(i,:)), round(curve(i+1,:)), map)
            coll = true; return;
        end
    end
end


%Buildings animation
function drawBuildings(map, alphaVal)
    if nargin < 2, alphaVal = 0.6; end
    CC = bwconncomp(map);  % group connected voxels into buildings

    % color set (repeats if > 6 buildings)
    colors = [0.8 0.1 0.1;   % red
              0.1 0.6 0.8;   % cyan-blue
              0.2 0.8 0.2;   % green
              0.9 0.7 0.1;   % yellow
              0.6 0.2 0.8;   % purple
              0.9 0.4 0.1];  % orange

    for i = 1:CC.NumObjects
        voxels = CC.PixelIdxList{i};
        [X,Y,Z] = ind2sub(size(map), voxels);

        % bounding box for each building
        xmin = min(X); xmax = max(X)+1;
        ymin = min(Y); ymax = max(Y)+1;
        zmin = min(Z); zmax = max(Z)+1;

        % vertices of the cuboid
        verts = [xmin ymin zmin;
                 xmax ymin zmin;
                 xmax ymax zmin;
                 xmin ymax zmin;
                 xmin ymin zmax;
                 xmax ymin zmax;
                 xmax ymax zmax;
                 xmin ymax zmax];

        % faces of the cuboid
        faces = [1 2 3 4;   % bottom
                 5 6 7 8;   % top
                 1 2 6 5;   % front
                 2 3 7 6;   % right
                 3 4 8 7;   % back
                 4 1 5 8];  % left

        % pick color for this building
        col = colors(mod(i-1, size(colors,1))+1, :);

        patch('Vertices', verts, 'Faces', faces, ...
              'FaceColor', col, 'FaceAlpha', alphaVal, ...
              'EdgeColor', 'k', 'EdgeAlpha', 0.2);
    end
end
