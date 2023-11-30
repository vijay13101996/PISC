module misc
        implicit none
        private
        public :: hess_compress
        public :: hess_expand
        public :: hess_mul_v
        public :: hess_mul
        contains 
                subroutine hess_compress(arr,N,ndim,nbeads,arr_out)
                        integer, intent(in) :: N,ndim,nbeads
                        real(8), intent(in) :: arr(nbeads,ndim,nbeads,ndim,N)
                        real(8), intent(inout) :: arr_out(ndim*nbeads,ndim*nbeads,N)
                        arr_out = reshape(arr,(/ndim*nbeads,ndim*nbeads,N/))
        
                end subroutine hess_compress

                subroutine hess_expand(arr,N,ndim,nbeads,arr_out)
                        integer, intent(in) :: N,ndim,nbeads
                        real(8), intent(in) :: arr(ndim*nbeads,ndim*nbeads,N)
                        real(8), intent(out) :: arr_out(nbeads,ndim,nbeads,ndim,N)

                        arr_out = reshape(arr,(/nbeads,nbeads,ndim,ndim,N/))
                        
                end subroutine hess_expand

                subroutine hess_mul_v(ddpot,N,ndim,nbeads,arr_i,arr_o,dt)
                        integer, intent(in) :: N,ndim,nbeads
                        real(8), intent(in) :: ddpot(nbeads,ndim,nbeads,ndim,N)
                        real(8), intent(in) :: arr_i(nbeads,ndim,nbeads,ndim,N)
                        real(8), intent(inout) :: arr_o(nbeads,ndim,nbeads,ndim,N)
                        real(8), intent(in) :: dt
                        real(8) :: ddpot_tmp(ndim*nbeads,ndim*nbeads,N)
                        real(8) :: arr_i_tmp(ndim*nbeads,ndim*nbeads,N)
                        real(8) :: arr_o_tmp(ndim*nbeads,ndim*nbeads,N)
                        integer :: i
                        ddpot_tmp = 0.0d0
                        arr_i_tmp = 0.0d0       
                        arr_o_tmp = 0.0d0
                        call hess_compress(ddpot,N,ndim,nbeads,ddpot_tmp)
                        call hess_compress(arr_i,N,ndim,nbeads,arr_i_tmp)
                        call hess_compress(arr_o,N,ndim,nbeads,arr_o_tmp)

                        !OMP PARALLEL DO PRIVATE(i)
                        do i = 1, N
                                !The transpose operation is required because the compressed hessian is stored in row-major order
                                arr_o_tmp(:,:,i) = transpose(arr_o_tmp(:,:,i))
                                arr_o_tmp(:,:,i) = arr_o_tmp(:,:,i) - &
                                matmul( transpose(ddpot_tmp(:,:,i)), transpose(arr_i_tmp(:,:,i)) )*dt
                                arr_o_tmp(:,:,i) = transpose(arr_o_tmp(:,:,i))
                        end do
                        !OMP END PARALLEL DO
                        call hess_expand(arr_o_tmp,N,ndim,nbeads,arr_o)

                end subroutine hess_mul_v

                subroutine hess_mul(ddpot,N,ndim,arr_in,arr_out,dt)
                        integer, intent(in) :: N,ndim
                        real(8), intent(in) :: ddpot(ndim,ndim,N)
                        real(8), intent(in) :: arr_in(ndim,ndim,N)
                        real(8), intent(inout) :: arr_out(ndim,ndim,N)
                        real(8), intent(in) :: dt
                        integer :: i
                        !real(8) :: tin, tout
                      
                        !!!OMP PARALLEL DO PRIVATE(i) SHARED(arr_out,arr_in,ddpot)
                        do i = 1, N
                        !arr_out(:,:,i) = transpose(arr_out(:,:,i))
                        !arr_out(:,:,i) = arr_out(:,:,i) - matmul( transpose(ddpot(:,:,i)), transpose(arr_in(:,:,i)) )*dt
                        !arr_out(:,:,i) = transpose(arr_out(:,:,i))
                        
                        ! ddpot and arr_in are multiplied in reverse order because they become 
                        ! 'transposed' when they are passed to the subroutine
                        call dgemm('N','N',ndim,ndim,ndim,-dt,arr_in(:,:,i)&
                                ,ndim,ddpot(:,:,i),ndim,1.0d0,arr_out(:,:,i),ndim)
                        
                        end do
                        !!!OMP END PARALLEL DO
                end subroutine hess_mul

end module misc
