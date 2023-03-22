% 2023-02-25 Oscar Lundin
% Planck function applied to star and Ganymede. Blackbody approximation
% models the reflected flux of each body.

clc;clear;

k = 1.380e-23; % Boltzmann constant Joule/Kelvin
h = 6.62607015e-34; % Planck constant Joule/Hertz
c = 299792458; % Speed of light in vacuum meters/second

% Planck's law for wavelength
B = @(w,T) ((2*h*c^2)/(w^5))/(exp(h*c/(w*k*T))-1);% w is wavelength, T is temperature in kelvin.

Tstar = 5700; % Assumed effective temperature of star GSPC P330-E (a G2V star).

% Wavelength range
wmin = 0;
wmax = 5.5e-6;

%% Iterate vectors

%spectGanymede = [];
spectStar = [];

for t = wmin:0.001e-6:wmax
    %spectGanymede = [spectGanymede; B(t,Tgan)];
    spectStar = [spectStar; B(t,Tstar)];
end

% figure(1)
% plot((wmin:0.001e-6:wmax)', spectGanymede) 
% title('Ganymede spectrum')
% grid on

figure(2)
plot((wmin:0.001e-6:wmax)', spectStar) 
title('Star spectrum')
grid on

%% Normalize for distance





