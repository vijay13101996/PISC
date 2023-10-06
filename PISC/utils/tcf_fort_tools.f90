module tcf_tools
	use omp_lib
	implicit none
	private
	public :: two_pt_3op_tcf
	public :: two_pt_2op_tcf
	contains
		subroutine two_pt_3op_tcf(A,B,C,tlen,N,ndim,Corr)
			integer, intent(in) :: tlen, N, ndim
			real(kind=8), dimension(tlen,N,ndim), intent(in) :: A,B,C
			real(kind=8), dimension(N) :: dp
			integer :: i,j,k,d
			!f2py real(kind=8), intent(in,out,copy) :: Corr
			real(kind=8), dimension(tlen,tlen), intent(inout) :: Corr
			Corr = 0.0d0
			dp = 0.0d0
			!print*, 'tlen, N, ndim', tlen, N, ndim, shape(B)
			!$OMP PARALLEL DO PRIVATE(i,dp)	
			do i=1,tlen
				!!$OMP PARALLEL DO PRIVATE(j)
				do j=1,tlen
					do k=1,N
						dp = 0.0d0
						do d=1,ndim
							dp(k) = dp(k) + C(j,k,d)*B(i,k,d)*A(1,k,d)
						end do
						Corr(i,j) = Corr(i,j) + dp(k)
					end do
					Corr(i,j) = Corr(i,j)/N
				end do
				!!$OMP END PARALLEL DO
			end do
			!$OMP END PARALLEL DO
		end subroutine two_pt_3op_tcf

		subroutine two_pt_2op_tcf(B,C,tlen,N,ndim,Corr)
			integer, intent(in) :: tlen, N, ndim
			real(kind=8), dimension(tlen,N,ndim), intent(in) :: B,C
			real(kind=8), dimension(N) :: dp
			integer :: i,j,k,d
			!f2py real(kind=8), intent(in,out,copy) :: Corr
			real(kind=8), dimension(tlen,tlen), intent(inout) :: Corr
			Corr = 0.0d0
			dp = 0.0d0
			!print*, 'shape', shape(B), shape(C)
			!$OMP PARALLEL DO PRIVATE(i, dp)	
			do i=1,tlen
				!!$OMP PARALLEL DO PRIVATE(j)
				do j=1,tlen
					do k=1,N
						dp = 0.0d0
						do d=1,ndim
							dp(k) = dp(k) + C(j,k,d)*B(i,k,d)
						end do
						Corr(i,j) = Corr(i,j) + dp(k)
						!print*, 'k, dpk, fort', k, dp(k)
					end do
					Corr(i,j) = Corr(i,j)/N
				end do
				!!$OMP END PARALLEL DO
			end do
			!$OMP END PARALLEL DO
		end subroutine two_pt_2op_tcf

end module tcf_tools
