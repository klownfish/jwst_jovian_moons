% 2023-02-25 Oscar Lundin
% Planck function applied to star and Ganymede. Blackbody approximation
% models the reflected flux of each body.

clc;clear;

k = 1.380e-23; % Boltzmann constant Joule/Kelvin
h = 6.62607015e-34; % Planck constant Joule/Hertz
c = 299792458; % Speed of light in vacuum meters/second

% Planck's law for wavelength
B = @(w,T) ((2*h*c^2)/(w^5))*(1/(exp(h*c/(w*k*T))-1));% w is wavelength, T is temperature in kelvin.

Tstar = 5700; % Assumed effective temperature of star GSPC P330-E (a G2V star).
Tgan = 130; % Assumed effective temperature of Ganymede.

% Wavelength range
wmin = 0;
wmax = 5.3e-6;

%% Iterate vectors

spectGanymede = [];
spectStar = [];

for t = wmin:0.001e-6:wmax
    spectGanymede = [spectGanymede; B(t,Tgan)];
    spectStar = [spectStar; B(t,Tstar)];
end

figure(1)
plot((wmin:0.001e-6:wmax)', spectGanymede) 
title('Ganymede spectrum')
xlabel('Wavelength (m)')
ylabel('Flux density (W*sr^-1*m^-3')
grid on

spectStar = spectStar*10^26*10^-9;
figure(2)
plot((wmin:0.001e-6:wmax)', spectStar) 
title('Star spectrum')
xlabel('Wavelength (m)')
ylabel('Flux density (W*sr^-1*m^-3')
grid on

% %% Normalize for distance
% 
% % r = 2631.2e3; % radius of Ganymede (NASA)
% % l = 740.699267e9; % distance Sun-Jupiter
% % 
% % thetaMax = atan(r/l);
% % solidAng = 2*pi*(1-cos(thetaMax)); % Approximate solid angle of Ganymede in relation to sun.
% % 
% % spectStarNorm = (solidAng/(4*pi))*spectStar;
% 
% Lsun2jup = 740.699267e9;  %Planetviewer
% Lsunradius = 695700e3; % International Astronomical Union standard
% solidAngSun = 1/(Lsunradius)^2;
% spectStarSolidAng
% spectStarNorm = spectStar*(Lsunradius/Lsun2jup)^2;
% figure(3)
% plot((wmin:0.001e-6:wmax)', spectStarNorm) 
% title('Normalized star spectrum')
% xlabel('Wavelength (m)')
% ylabel('Flux density (W*sr^-1*m^-3')
% grid on
% 
% 
% 

%% New normalization

Lsun2jup = 740.699267e9;  %Planetviewer
Lsunradius = 695700e3; % International Astronomical Union standard

areaSun = (4*pi*Lsunradius^3)/3;
areaSphereGan = (4*pi*Lsun2jup^3);


totalPowerPerUnitArea = spectStar*(areaSun/areaSphereGan)*(Lsunradius/Lsun2jup)^2;

figure(3)
plot((wmin:0.001e-6:wmax)', totalPowerPerUnitArea) 
title('Normalized star spectrum')
xlabel('Wavelength (m)')
ylabel('Power per unit area and wavelength W/(m^3)');
grid on








