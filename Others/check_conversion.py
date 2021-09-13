import matplotlib.pyplot as plt

original = [10.000000, 12.429802, 14.713967, 16.715590, 18.314697, 19.415440, 19.951847, 19.891766, 19.238796, 18.032074, 16.343933, 14.275549, 11.950905, 9.509323, 7.097153, 4.858975, 2.928932, 1.422712, 0.430596, 0.012046, 0.192147, 0.960107, 2.269898, 4.043004, 6.173162, 8.532696, 10.980171, 13.368898, 15.555702, 17.409512, 18.819210, 19.700314, 20.000000, 19.700314, 18.819210, 17.409512, 15.555702, 13.368898, 10.980179, 8.532696, 6.173166, 4.043013, 2.269896, 0.960107, 0.192145, 0.012046, 0.430594, 1.422714, 2.928927, 4.858973, 7.097153, 9.509331, 11.950903, 14.275537, 16.343933, 18.032074, 19.238796, 19.891766, 19.951847, 19.415440, 18.314705, 16.715590, 14.713967, 12.429802, 10.000000, 7.570212, 5.286033, 3.284410, 1.685296, 0.584559, 0.048153, 0.108233, 0.761205, 1.967925, 3.656067, 5.724449, 8.049082, 10.490677, 12.902847, 15.141014, 17.071068, 18.577286, 19.569399, 19.987953, 19.807854, 19.039900, 17.730104, 15.956993, 13.826820, 11.467304, 9.019829, 6.631087, 4.444310, 2.590499, 1.180788, 0.299687, 0.000000, 0.299687, 1.180788, 2.590489, 4.444298, 6.631102, 9.019843, 11.467275, 13.826834, 15.956993, 17.730085, 19.039894, 19.807854, 19.987957, 19.569405, 18.577286, 17.071068, 15.141027, 12.902847, 10.490677, 8.049097, 5.724449, 3.656067, 1.967907, 0.761216, 0.108234, 0.048153, 0.584559, 1.685304, 3.284410, 5.286033, 7.570198, 10.000000, 12.429802, 14.713942, 16.715567, 18.314697, 19.415440, 19.951847, 19.891766, 19.238785, 18.032074, 16.343933, 14.275551, 11.950903, 9.509354, 7.097182, 4.858973, 2.928932, 1.422714, 0.430596, 0.012044, 0.192147, 0.960107, 2.269896, 4.042983, 6.173138, 8.532665, 10.980171, 13.368898, 15.555702, 17.409512, 18.819199, 19.700306, 20.000000, 19.700314, 18.819212, 17.409531, 15.555727, 13.368898, 10.980171, 8.532696, 6.173166, 4.042983, 2.269915, 0.960107, 0.192147, 0.012046, 0.430596, 1.422729, 2.928953, 4.858973, 7.097153, 9.509323, 11.950903, 14.275578, 16.343956, 18.032093, 19.238785, 19.891760, 19.951851, 19.415440, 18.314697, 16.715590, 14.713967, 12.429831, 10.000030, 7.570198, 5.286033, 3.284410, 1.685304, 0.584550, 0.048153, 0.108234, 0.761205, 1.967925, 3.656067, 5.724477, 8.049126, 10.490617, 12.902789, 15.141027, 17.071068, 18.577286, 19.569405, 19.987953, 19.807865, 19.039894, 17.730104, 15.956993, 13.826834, 11.467304, 9.019888, 6.631102, 4.444298, 2.590489, 1.180788, 0.299687, 0.000000, 0.299687, 1.180788, 2.590489, 4.444298, 6.631102, 9.019829, 11.467304, 13.826834, 15.956993, 17.730104, 19.039894, 19.807854, 19.987953, 19.569386, 18.577316, 17.071110, 15.141027, 12.902847, 10.490677, 8.049097, 5.724449, 3.656067, 1.967925, 0.761205, 0.108234, 0.048153, 0.584559, 1.685304, 3.284410, 5.286033, 7.570198]

mcu = [9.999998, 12.429802, 14.713968, 16.715591, 18.314697, 19.415440, 19.951847, 19.891766, 19.238794, 18.032072, 16.343933, 14.275549, 11.950904, 9.509325, 7.097151, 4.858974, 2.928931, 1.422715, 0.430593, 0.012044, 0.192147, 0.960107, 2.269896, 4.043005, 6.173162, 8.532697, 10.980170, 13.368897, 15.555702, 17.409512, 18.819208, 19.700314, 20.000000, 19.700314, 18.819210, 17.409512, 15.555702, 13.368899, 10.980178, 8.532696, 6.173164, 4.043015, 2.269894, 0.960106, 0.192145, 0.012047, 0.430592, 1.422714, 2.928927, 4.858973, 7.097151, 9.509331, 11.950905, 14.275539, 16.343931, 18.032074, 19.238796, 19.891764, 19.951847, 19.415440, 18.314705, 16.715588, 14.713966, 12.429803, 10.000000, 7.570213, 5.286033, 3.284411, 1.685294, 0.584560, 0.048150, 0.108233, 0.761203, 1.967928, 3.656065, 5.724448, 8.049083, 10.490679, 12.902845, 15.141013, 17.071068, 18.577284, 19.569397, 19.987953, 19.807856, 19.039900, 17.730106, 15.956993, 13.826818, 11.467305, 9.019827, 6.631086, 4.444308, 2.590502, 1.180786, 0.299687, -0.000001, 0.299686, 1.180783, 2.590487, 4.444297, 6.631103, 9.019843, 11.467275, 13.826832, 15.956993, 17.730085, 19.039894, 19.807854, 19.987959, 19.569405, 18.577286, 17.071068, 15.141027, 12.902847, 10.490677, 8.049095, 5.724451, 3.656065, 1.967906, 0.761214, 0.108238, 0.048150, 0.584558, 1.685303, 3.284413, 5.286030, 7.570200, 9.999998, 12.429802, 14.713943, 16.715569, 18.314697, 19.415440, 19.951847, 19.891766, 19.238783, 18.032072, 16.343933, 14.275551, 11.950902, 9.509356, 7.097180, 4.858972, 2.928931, 1.422717, 0.430593, 0.012042, 0.192147, 0.960107, 2.269893, 4.042984, 6.173138, 8.532666, 10.980170, 13.368897, 15.555702, 17.409512, 18.819197, 19.700306, 20.000000, 19.700314, 18.819214, 17.409531, 15.555725, 13.368899, 10.980170, 8.532695, 6.173164, 4.042984, 2.269913, 0.960106, 0.192147, 0.012047, 0.430593, 1.422729, 2.928954, 4.858973, 7.097151, 9.509323, 11.950905, 14.275578, 16.343954, 18.032093, 19.238785, 19.891758, 19.951851, 19.415440, 18.314697, 16.715588, 14.713966, 12.429831, 10.000031, 7.570199, 5.286]

fig, axs = plt.subplots(2)
axs[0].plot(original)
axs[0].set_title('Original')
axs[1].plot(mcu)
axs[1].set_title('Converted')
plt.tight_layout()
plt.show()