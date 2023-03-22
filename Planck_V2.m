% 2023-03-09 Oscar Lundin
% Planck function applied to star and Ganymede. Blackbody approximation
% models the reflected flux of each body.

clc;clear;

k = 1.380e-23; % Boltzmann constant Joule/Kelvin
h = 6.62607015e-34; % Planck constant Joule/Hertz
c = 299792458; % Speed of light in vacuum meters/second

% Planck's law for wavelength
%B = @(w,T) ((2*h*c^2)/(w^5))*(1/(exp(h*c/(w*k*T))-1));% w is wavelength, T is temperature in kelvin.

% Planck's law for frequency
B = @(f,T) ( 2*h*f^3 )/( c^2 ) * 1/( exp((h*f)/(k*T)) - 1 );
Tstar = 5700; % Assumed effective temperature of star GSPC P330-E (a G2V star).
Tgan = 130; % Assumed effective temperature of Ganymede.
% Wavelength range
wmin = 0.6e-6;
wmax = 5.3e-6;

%% Iterate vectors

FluxDensityGanymede = [];
FluxDensityStar = [];

for t = wmin:0.001e-6:wmax
    FluxDensityGanymede = [FluxDensityGanymede; B(c/t,Tgan)];
    FluxDensityStar = [FluxDensityStar; B(c/t,Tstar)];
end

figure(1)
plot((wmin:0.001e-6:wmax)', FluxDensityGanymede) 
title('Ganymede spectrum')
xlabel('Wavelength (m)')
ylabel('Flux density 10^26 Jy/sr')
grid on

FluxDensityStar = FluxDensityStar*10^26*10^-9;
figure(2)
plot((wmin:0.001e-6:wmax)', FluxDensityStar) 
title('Star spectrum')
xlabel('Wavelength (m)')
ylabel('Flux density 10^(26) Jy/sr')
grid on

%% Power per unit area from Sun to Ganymede

Lsun2jup = 740.699267e9;  %Planetviewer
Lsunradius = 695700e3; % International Astronomical Union standard

areaSun = (4*pi*Lsunradius^3)/3;
areaSphereGan = (4*pi*Lsun2jup^3);


totalPowerPerUnitArea = FluxDensityStar*(areaSun/areaSphereGan)*(Lsunradius/Lsun2jup)^2;

figure(3)
plot((wmin:0.001e-6:wmax)', totalPowerPerUnitArea) 
title('Power from sun on Ganymede')
xlabel('Wavelength (m)')
ylabel('Power per unit area W/m^2');
grid on



