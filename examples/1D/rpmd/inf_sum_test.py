import numpy as np

m = 1.0
beta= 1.

omega = 1.

mu0 = 1/(2*m*omega*np.sinh(beta*omega/2))
mu2 = omega**2*mu0
mu4 = omega**4*mu0
mu6 = omega**6*mu0
mu8 = omega**8*mu0

k_thres = 2
k_term = np.arange(1,k_thres+1)

mats_freqs = 2*np.pi*k_term/beta
odd_freqs = mats_freqs[0::2]
even_freqs = mats_freqs[1::2]

inf_series_odd =  np.sum( 1/(mats_freqs[0::2]**2 + omega**2))/(m*beta)
inf_series_even = np.sum( 1/(mats_freqs[1::2]**2 + omega**2))/(m*beta)

zero_term = 1/(m*omega**2*beta)

mu_0_sum = zero_term - 2*inf_series_odd + 2*inf_series_even

inf2_series_odd =  np.sum(odd_freqs**2/(odd_freqs**2 + omega**2))/(m*beta)
inf2_series_even = np.sum(even_freqs**2/(even_freqs**2 + omega**2))/(m*beta)

mu_2_sum =  2*inf2_series_odd - 2*inf2_series_even + 1/(m*beta)

inf4_series_odd =  np.sum(odd_freqs**4/(odd_freqs**2 + omega**2))/(m*beta)
inf4_series_even = np.sum(even_freqs**4/(even_freqs**2 + omega**2))/(m*beta)

mu_4_sum = (-2*inf4_series_odd + 2*inf4_series_even) #- 2*(np.sum(mats_freqs[1::2]**2 - mats_freqs[0::2]**2))/(m*beta) + omega**2/(m*beta)

inf6_series_odd =  np.sum(odd_freqs**6/(odd_freqs**2 + omega**2))/(m*beta)
inf6_series_even = np.sum(even_freqs**6/(even_freqs**2 + omega**2))/(m*beta)

mu_6_sum = 2*inf6_series_odd - 2*inf6_series_even + 2*(np.sum(mats_freqs[1::2]**4 - mats_freqs[0::2]**4))/(m*beta) \
            -2*omega**2*(np.sum(mats_freqs[1::2]**2 - mats_freqs[0::2]**2))/(m*beta) + omega**4/(m*beta)

inf8_series_odd =  np.sum(odd_freqs**8/(odd_freqs**2 + omega**2))/(m*beta)
inf8_series_even = np.sum(even_freqs**8/(even_freqs**2 + omega**2))/(m*beta)

mu_8_sum = -2*inf8_series_odd + 2*inf8_series_even - 2*(np.sum(mats_freqs[1::2]**6 - mats_freqs[0::2]**6))/(m*beta) +\
            2*omega**2*(np.sum(mats_freqs[1::2]**4 - mats_freqs[0::2]**4))/(m*beta) - 2*omega**4*(np.sum(mats_freqs[1::2]**2 - mats_freqs[0::2]**2))/(m*beta) + omega**6/(m*beta)

print('mu0 = ', mu0, 'mu0_sum = ', mu_0_sum)
print('mu2 = ', mu2, 'mu2_sum = ', mu_2_sum)
print('mu4 = ', mu4, 'mu4_sum = ', mu_4_sum)
print('mu6 = ', mu6, 'mu6_sum = ', mu_6_sum)
print('mu8 = ', mu8, 'mu8_sum = ', mu_8_sum)
