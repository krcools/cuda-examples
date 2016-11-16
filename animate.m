function F = animate(data)

[nt,nx] = size(data);



figure();
hold on;
x = linspace(0.0, 1.0, nx);
M = max(data(:));
axis([0.0 1.0 -M M]);
F(nt) = struct('cdata',[],'colormap',[]);
for i = 1:nt
    h = plot(x, data(i,:),'-k.','LineWidth',2);
    F(i) = getframe();
    delete(h);
end