module harmonic_oblique
        implicit none
        private
        public :: potential, dpotential, ddpotential
        public :: potential_f, dpotential_f, ddpotential_f
        contains
        subroutine potential(q,ndim,param_list,lp,Vq)
                integer, intent(in) :: ndim, lp
                real(8), intent(in) :: q(ndim)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(out) :: Vq
                real(8) :: x,y,m,omega1,omega2,T11,T12,T21,T22,xi,eta

                m      = param_list(1) 
                omega1 = param_list(2)
                omega2 = param_list(3)
                T11    = param_list(4)
                T12    = param_list(5)
                T21    = param_list(6)
                T22    = param_list(7)
                
                x = q(1)
                y = q(2)

                xi  = T11*x + T12*y
                eta = T21*x + T22*y
                Vq = 0.5*m*omega1**2*xi**2 + 0.5*m*omega2**2*eta**2

        end subroutine potential

        subroutine dpotential(q,ndim,param_list,lp,dVq)
                integer, intent(in) :: ndim, lp
                real(8), intent(in) :: q(ndim)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: dVq(ndim)
                real(8) :: x,y,m,omega1,omega2,T11,T12,T21,T22,xi,eta

                m      = param_list(1) 
                omega1 = param_list(2)
                omega2 = param_list(3)
                T11    = param_list(4)
                T12    = param_list(5)
                T21    = param_list(6)
                T22    = param_list(7)
                
                x = q(1)
                y = q(2)

                xi  = T11*x + T12*y
                eta = T21*x + T22*y
                 
                dVq(1) = T11*m*omega1**2*xi + T21*m*omega2**2*eta
                dVq(2) = T12*m*omega1**2*xi + T22*m*omega2**2*eta

        end subroutine dpotential

        subroutine ddpotential(q,ndim,param_list,lp,ddVq)
                integer, intent(in) :: ndim, lp
                real(8), intent(in) :: q(ndim)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: ddVq(ndim,ndim)
                real(8) :: x,y,m,omega1,omega2,T11,T12,T21,T22,xi,eta

                m      = param_list(1) 
                omega1 = param_list(2)
                omega2 = param_list(3)
                T11    = param_list(4)
                T12    = param_list(5)
                T21    = param_list(6)
                T22    = param_list(7)
                
                x = q(1)
                y = q(2)

                xi  = T11*x + T12*y
                eta = T21*x + T22*y

                ddVq(1,1) = T11**2*m*omega1**2 + T21**2*m*omega2**2
                ddVq(2,2) = T12**2*m*omega1**2 + T22**2*m*omega2**2
                ddVq(1,2) = T11*T12*m*omega1**2 + T21*T22*m*omega2**2
                ddVq(2,1) = ddVq(1,2)

        end subroutine ddpotential
 
        subroutine potential_f(qcart,N,ndim,nbeads,param_list,lp,pot)
                integer, intent(in) :: N, ndim, nbeads, lp
                real(8), intent(in) :: qcart(nbeads,ndim,N)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: pot(nbeads,N)
                integer :: i,j

                !$omp parallel do private(i,j) shared(qcart,pot)
                do i = 1, N
                        do j = 1, nbeads
                                call potential(qcart(j,:,i),ndim,param_list,lp,pot(j,i))
                        end do
                end do
                !$omp end parallel do

        end subroutine potential_f

        subroutine dpotential_f(qcart,N,ndim,nbeads,param_list,lp,dpot_cart)
                integer, intent(in) :: N, ndim, nbeads, lp
                real(8), intent(in) :: qcart(nbeads,ndim,N)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: dpot_cart(nbeads,ndim,N)
                integer :: i,j

                !$omp parallel do private(i,j) shared(qcart,dpot_cart)
                do i = 1, N
                        do j = 1, nbeads
                                call dpotential(qcart(j,:,i),ndim,param_list,lp,dpot_cart(j,:,i))
                        end do
                end do
                !$omp end parallel do

        end subroutine dpotential_f
                
        subroutine ddpotential_f(qcart,N,ndim,nbeads,param_list,lp,ddpot_cart)
                integer, intent(in) :: N, ndim, nbeads, lp
                real(8), intent(in) :: qcart(nbeads,ndim,N)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: ddpot_cart(nbeads,ndim,nbeads,ndim,N)
                integer :: i,j

                !$omp parallel do private(i,j) shared(qcart,ddpot_cart)
                do i=1, N
                        do j=1, nbeads
                                call ddpotential(qcart(j,:,i),ndim,param_list,lp,ddpot_cart(j,:,j,:,i))
                        end do
                end do
                !$omp end parallel do

        end subroutine ddpotential_f
end module harmonic_oblique

