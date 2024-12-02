function [x,y,dx,dy,epsr] = get_mesh(structure,lambda0, wg_width_bottom, wg_height, gap_left, gap_right, gap_top, ...
    gap_bottom,  ppw, plot_geo, n_background, n_wg,n_substrate, n_film, film_thickness, wg_width_top)

nmax = max([n_wg, n_background]);
dmesh = round(lambda0*1e9/ppw/nmax)*1e-9;

Nr_wg = round(wg_width_bottom/dmesh);      % for tapered, this is the bottom width
dwg = wg_width_bottom/Nr_wg;
Nr_leftgap = round(gap_left/dmesh);
dgleft = gap_left/Nr_leftgap;
Nr_rightgap = round(gap_right/dmesh);
dgright = gap_right/Nr_rightgap;
Nz_topgap = round(gap_top/dmesh);
dgtop = gap_top/Nz_topgap;
Nz_bottomgap = round(gap_bottom/dmesh);
dgbottom = gap_bottom/Nz_bottomgap;
Nz_wg = round(wg_height/dmesh);
dz = wg_height/Nz_wg;

switch structure
    case 'wg_only'               % waveguide only
        Nz = Nz_bottomgap+Nz_topgap+Nz_wg;
        Nr = Nr_wg+Nr_leftgap+Nr_rightgap;

        x = cumsum([ ones(1,Nr_leftgap)*dgleft ones(1,Nr_wg)*dwg ones(1,Nr_rightgap)*dgright  ]);
        y = cumsum([ ones(1,Nz_bottomgap)*dgbottom ones(1,Nz_wg)*dz ones(1,Nz_topgap)*dgtop ]);
        x = x-mean(x);
        y = y-mean(y);

        dx = diff(x); dx = [dx dx(end)];
        dy = diff(y); dy= [dy dy(end)];

        epsr = ones(Nz,Nr)*n_background^2;
        epsr(Nz_bottomgap+1:Nz_bottomgap+Nz_wg, Nr_leftgap+1:Nr_leftgap+Nr_wg) = n_wg^2; % waveguide

    case 'tapered_wg_only'               % waveguide only
        Nz = Nz_bottomgap+Nz_topgap+Nz_wg;
        Nr = Nr_wg+Nr_leftgap+Nr_rightgap;

        x = cumsum([ ones(1,Nr_leftgap)*dgleft ones(1,Nr_wg)*dwg ones(1,Nr_rightgap)*dgright  ]);
        y = cumsum([ ones(1,Nz_bottomgap)*dgbottom ones(1,Nz_wg)*dz ones(1,Nz_topgap)*dgtop ]);
        x = x-mean(x);
        y = y-Nz_bottomgap*dgbottom; % for tapered, low boundary of the WG has to be at y=0       

        dx = diff(x); dx = [dx dx(end)];
        dy = diff(y); dy= [dy dy(end)];

        epsr = ones(Nz,Nr)*n_background^2;
     
        [XX, YY] = meshgrid(x,y);
        px1 = wg_width_bottom/2;
        py1 =  0;
        px2 =  wg_width_top/2;
        py2 = wg_height;
        px3 = -px2;
        py3 = py2;
        px4 = -px1;
        py4 = py1;
        % figure; plot([px1 px2 px3 px4 px1],[py1 py2 py3 py4 py1])

        % idea: if the sum of areas of 4 triangles is less than the area of the
        % trapezoid, then the point is inside

        Area  = 0;
        Area = Area + 0.5* abs( XX.*(py1-py2)+px1.*(py2-YY) + px2.*(YY-py1) );
        Area = Area + 0.5* abs( XX.*(py2-py3)+px2.*(py3-YY) + px3.*(YY-py2) );
        Area = Area + 0.5* abs( XX.*(py3-py4)+px3.*(py4-YY) + px4.*(YY-py3) );
        Area = Area + 0.5* abs( XX.*(py4-py1)+px4.*(py1-YY) + px1.*(YY-py4) );
        trapezoid_area = (wg_width_bottom+wg_width_top)/2*wg_height;
        epsr(Area<=trapezoid_area*1.01) = n_wg^2;

    case 'wg_on_substrate'
        Nz = Nz_bottomgap+Nz_topgap+Nz_wg;
        Nr = Nr_wg+Nr_leftgap+Nr_rightgap;

        x = cumsum([ ones(1,Nr_leftgap)*dgleft ones(1,Nr_wg)*dwg ones(1,Nr_rightgap)*dgright  ]);
        y = cumsum([ ones(1,Nz_bottomgap)*dgbottom ones(1,Nz_wg)*dz ones(1,Nz_topgap)*dgtop ]);
        x = x-mean(x);
        y = y-Nz_bottomgap*dgbottom;

        dx = diff(x); dx = [dx dx(end)];
        dy = diff(y); dy= [dy dy(end)];

        epsr = ones(Nz,Nr)*n_background^2;
        epsr(Nz_bottomgap+1:Nz_bottomgap+Nz_wg, Nr_leftgap+1:Nr_leftgap+Nr_wg) = n_wg^2; % waveguide
        epsr(1:Nz_bottomgap, :) = n_substrate^2; % substrate

    case 'tapered_wg_on_substrate'
        Nz = Nz_bottomgap+Nz_topgap+Nz_wg;
        Nr = Nr_wg+Nr_leftgap+Nr_rightgap;

        x = cumsum([ ones(1,Nr_leftgap)*dgleft ones(1,Nr_wg)*dwg ones(1,Nr_rightgap)*dgright  ]);
        y = cumsum([ ones(1,Nz_bottomgap)*dgbottom ones(1,Nz_wg)*dz ones(1,Nz_topgap)*dgtop ]);
        x = x-mean(x);
        y = y-Nz_bottomgap*dgbottom;

        dx = diff(x); dx = [dx dx(end)];
        dy = diff(y); dy= [dy dy(end)];

        epsr = ones(Nz,Nr)*n_background^2;        
        epsr(1:Nz_bottomgap, :) = n_substrate^2; % substrate

        [XX, YY] = meshgrid(x,y);
        px1 = wg_width_bottom/2;
        py1 =  0;
        px2 =  wg_width_top/2;
        py2 = wg_height;
        px3 = -px2;
        py3 = py2;
        px4 = -px1;
        py4 = py1;
        % figure; plot([px1 px2 px3 px4 px1],[py1 py2 py3 py4 py1])

        % idea: if the sum of areas of 4 triangles is less than the area of the
        % trapezoid, then the point is inside

        Area  = 0;
        Area = Area + 0.5* abs( XX.*(py1-py2)+px1.*(py2-YY) + px2.*(YY-py1) );
        Area = Area + 0.5* abs( XX.*(py2-py3)+px2.*(py3-YY) + px3.*(YY-py2) );
        Area = Area + 0.5* abs( XX.*(py3-py4)+px3.*(py4-YY) + px4.*(YY-py3) );
        Area = Area + 0.5* abs( XX.*(py4-py1)+px4.*(py1-YY) + px1.*(YY-py4) );
        trapezoid_area = (wg_width_bottom+wg_width_top)/2*wg_height;
        epsr(Area<=trapezoid_area*1.01) = n_wg^2;         

    case 'wg_on_film_and_substrate'
        Nz_film = round(film_thickness/dmesh);
        dgfilm= film_thickness/Nz_film;

        Nz = Nz_bottomgap+Nz_film+Nz_wg+Nz_topgap;
        Nr = Nr_wg+Nr_leftgap+Nr_rightgap;

        x = cumsum([ ones(1,Nr_leftgap)*dgleft ones(1,Nr_wg)*dwg ones(1,Nr_rightgap)*dgright  ]);
        y = cumsum([ ones(1,Nz_bottomgap)*dgbottom ones(1,Nz_film)*dgfilm  ones(1,Nz_wg)*dz ones(1,Nz_topgap)*dgtop ]);
        x = x-mean(x);
        y = y-Nz_bottomgap*dgbottom-Nz_film*dgfilm;

        dx = diff(x); dx = [dx dx(end)];
        dy = diff(y); dy= [dy dy(end)];

        epsr = ones(Nz,Nr)*n_background^2;
        epsr(Nz_bottomgap+Nz_film+1:Nz_bottomgap+Nz_film+Nz_wg, Nr_leftgap+1:Nr_leftgap+Nr_wg) = n_wg^2; % waveguide

        epsr(Nz_bottomgap+1:Nz_bottomgap+Nz_film, :) = n_film^2;        % thin film
        epsr(1:Nz_bottomgap, :) = n_substrate^2;                                       % substrate
    case 'tapered_wg_on_film_and_substrate'
        Nz_film = round(film_thickness/dmesh);
        dgfilm= film_thickness/Nz_film;

        Nz = Nz_bottomgap+Nz_film+Nz_wg+Nz_topgap;
        Nr = Nr_wg+Nr_leftgap+Nr_rightgap;

        x = cumsum([ ones(1,Nr_leftgap)*dgleft ones(1,Nr_wg)*dwg ones(1,Nr_rightgap)*dgright  ]);
        y = cumsum([ ones(1,Nz_bottomgap)*dgbottom ones(1,Nz_film)*dgfilm  ones(1,Nz_wg)*dz ones(1,Nz_topgap)*dgtop ]);
        x = x-mean(x);
        y = y-Nz_bottomgap*dgbottom-Nz_film*dgfilm;

        dx = diff(x); dx = [dx dx(end)];
        dy = diff(y); dy= [dy dy(end)];

        epsr = ones(Nz,Nr)*n_background^2;
        epsr(Nz_bottomgap+1:Nz_bottomgap+Nz_film, :) = n_film^2;        % thin film
        epsr(1:Nz_bottomgap, :) = n_substrate^2;                                       % substrate        

        [XX, YY] = meshgrid(x,y);
        px1 = wg_width_bottom/2;
        py1 =  0;
        px2 =  wg_width_top/2;
        py2 = wg_height;
        px3 = -px2;
        py3 = py2;
        px4 = -px1;
        py4 = py1;
        % figure; plot([px1 px2 px3 px4 px1],[py1 py2 py3 py4 py1])

        % idea: if the sum of areas of 4 triangles is less than the area of the
        % trapezoid, then the point is inside

        Area  = 0;
        Area = Area + 0.5* abs( XX.*(py1-py2)+px1.*(py2-YY) + px2.*(YY-py1) );
        Area = Area + 0.5* abs( XX.*(py2-py3)+px2.*(py3-YY) + px3.*(YY-py2) );
        Area = Area + 0.5* abs( XX.*(py3-py4)+px3.*(py4-YY) + px4.*(YY-py3) );
        Area = Area + 0.5* abs( XX.*(py4-py1)+px4.*(py1-YY) + px1.*(YY-py4) );
        trapezoid_area = (wg_width_bottom+wg_width_top)/2*wg_height;
        epsr(Area<=trapezoid_area*1.01) = n_wg^2;        
    otherwise
        disp(structure)
        disp('check your structure type')
end


if plot_geo ==1
    figure('position',[100 500 700 500])
    h = pcolor(x*1e6,y*1e6,abs(sqrt(epsr)));
    xlabel('x (\mum)')
    ylabel('y (\mum)')
    colorbar
    axis(1e6*[min(x) max(x) min(y) max(y)])
    colormap(flipud(bone))
    clim([min(min(sqrt(epsr))) max(max(sqrt(epsr)))*1.5])
end