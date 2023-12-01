module quartic_bistable
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
                real(8) :: x,y, alpha, D, lamda, g, z
                real(8) :: ea, eay, quartx, vy, vx, vxy
                        
                alpha = param_list(1)
                D = param_list(2)
                lamda = param_list(3)
                g = param_list(4)
                z = param_list(5)

                x = q(1)
                y = q(2)
                
                eay = exp(-alpha*y)
                quartx = (x**2 - lamda**2/(8*g))
                vy = D*(1-eay)**2
                vx = g*quartx**2
                vxy = (vx-lamda**4/(64*g))*(exp(-z*alpha*y) - 1)

                Vq = vx + vy + vxy

        end subroutine potential

        subroutine dpotential(q,ndim,param_list,lp,dVq)
                integer, intent(in) :: ndim, lp
                real(8), intent(in) :: q(ndim)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: dVq(ndim)
                real(8) :: x,y, alpha, D, lamda, g, z
                real(8) :: ea, eay, quartx, vy, vx, vxy
                        
                alpha = param_list(1)
                D = param_list(2)
                lamda = param_list(3)
                g = param_list(4)
                z = param_list(5)

                x = q(1)
                y = q(2)
                
                dVq(1) = 4*g*x*(-1 + exp(-alpha*y*z))*(x**2 - lamda**2/(8*g)) + 4*g*x*(x**2 - lamda**2/(8*g)) 
                dVq(2) = 2*D*alpha*(1 - exp(-alpha*y))*exp(-alpha*y) - &
                        alpha*z*(g*(x**2 - lamda**2/(8*g))**2 - lamda**4/(64*g))*exp(-alpha*y*z) 

        end subroutine dpotential

        subroutine ddpotential(q,ndim,param_list,lp,ddVq)
                integer, intent(in) :: ndim, lp
                real(8), intent(in) :: q(ndim)
                real(8), intent(in) :: param_list(lp)
                real(8), intent(inout) :: ddVq(ndim,ndim)
                real(8) :: x,y, alpha, D, lamda, g, z
                real(8) :: ea, eay, quartx, vy, vx, vxy
                        
                alpha = param_list(1)
                D = param_list(2)
                lamda = param_list(3)
                g = param_list(4)
                z = param_list(5)
                
                x = q(1)
                y = q(2)
                
                ddVq(1,1) = 4*g*(-2*x**2*(1 - exp(-alpha*y*z)) + 3*x**2 - &
                            (1 - exp(-alpha*y*z))*(8*x**2 - lamda**2/g)/8 - lamda**2/(8*g))
                ddVq(1,2) = -4*alpha*g*x*z*(x**2 - lamda**2/(8*g))*exp(-alpha*y*z) 
                ddVq(2,2) = alpha**2*(-2*D*(1 - exp(-alpha*y))*exp(-alpha*y) + 2*D*exp(-2*alpha*y) &
                            + z**2*(g*(8*x**2 - lamda**2/g)**2 - lamda**4/g)*exp(-alpha*y*z)/64) 
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
end module quartic_bistable
