module henon_heiles
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
                real(8) :: x,y, lamda, g
                real(8) :: vh,vp,vs
                
                lamda = param_list(1)
                g = param_list(2)

                vh = 0.5*(x**2+y**2)
                vp = lamda*(x**2*y - y**3/3.0)
                vs = g*(x**2+y**2)**2  
                
                x = q(1)
                y = q(2)
                
                Vq = vh + vp + vs

        end subroutine potential

        subroutine dpotential(q,ndim,param_list,lp,dVq)
                integer, intent(in) :: ndim, lp
                real(8), intent(in) :: q(ndim)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: dVq(ndim)
                real(8) :: x,y, lamda, g
                        
                lamda = param_list(1)
                g = param_list(2)

                x = q(1)
                y = q(2)
                
                dVq(1) = 4*g*x*(x**2 + y**2) + 2*lamda*x*y + x
                dVq(2) = 4*g*y*(x**2 + y**2) + lamda*(x**2 - y**2) + y

        end subroutine dpotential

        subroutine ddpotential(q,ndim,param_list,lp,ddVq)
                integer, intent(in) :: ndim, lp
                real(8), intent(in) :: q(ndim)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: ddVq(ndim,ndim)
                real(8) :: x,y, lamda, g
                         
                x = q(1)
                y = q(2)
                
                lamda = param_list(1)
                g = param_list(2)


                ddVq(1,1) = 8*g*x**2 + 4*g*(x**2 + y**2) + 2*lamda*y + 1.0
                ddVq(1,2) = 2*x*(4*g*y + lamda)
                ddVq(2,2) = 8*g*y**2 + 4*g*(x**2 + y**2) - 2.0*lamda*y + 1.0
                ddVq(2,1) = ddVq(1,2)

        end subroutine ddpotential
 
        subroutine potential_f(qcart,N,ndim,nbeads,param_list,lp,pot)
                integer, intent(in) :: N, ndim, nbeads, lp
                real(8), intent(in) :: qcart(nbeads,ndim,N)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: pot(nbeads,N)
                integer :: i,j

                !!$omp parallel do private(i,j) shared(qcart,pot)
                do i = 1, N
                        do j = 1, nbeads
                                call potential(qcart(j,:,i),ndim,param_list,lp,pot(j,i))
                        end do
                end do
                !!$omp end parallel do

        end subroutine potential_f

        subroutine dpotential_f(qcart,N,ndim,nbeads,param_list,lp,dpot_cart)
                integer, intent(in) :: N, ndim, nbeads, lp
                real(8), intent(in) :: qcart(nbeads,ndim,N)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: dpot_cart(nbeads,ndim,N)
                integer :: i,j

                !!$omp parallel do private(i,j) shared(qcart,dpot_cart)
                do i = 1, N
                        do j = 1, nbeads
                                call dpotential(qcart(j,:,i),ndim,param_list,lp,dpot_cart(j,:,i))
                        end do
                end do
                !!$omp end parallel do

        end subroutine dpotential_f
                
        subroutine ddpotential_f(qcart,N,ndim,nbeads,param_list,lp,ddpot_cart)
                integer, intent(in) :: N, ndim, nbeads, lp
                real(8), intent(in) :: qcart(nbeads,ndim,N)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: ddpot_cart(nbeads,ndim,nbeads,ndim,N)
                integer :: i,j

                !!$omp parallel do private(i,j) shared(qcart,ddpot_cart)
                do i=1, N
                        do j=1, nbeads
                                call ddpotential(qcart(j,:,i),ndim,param_list,lp,ddpot_cart(j,:,j,:,i))
                        end do
                end do
                !!$omp end parallel do

        end subroutine ddpotential_f
end module henon_heiles
