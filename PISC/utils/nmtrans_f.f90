
module nmtrans
        use omp_lib
        implicit none
        private
        public :: dgemm_matmul
        public :: cart2mats
        public :: mats2cart
        public :: cart2mats_hessian
        public :: mats2cart_hessian


        contains
                function dgemm_matmul(A,B,nd) result(C)
                        integer, intent(in) :: nd
                        real(8), intent(in) :: A(nd,nd), B(nd,nd)
                        real(8) :: C(nd,nd)
                        C(:,:) = 0.0d0
                        call dgemm('N','N',nd,nd,nd,1.0d0,A,nd,B,nd,0.0d0,C,nd)
                end function dgemm_matmul

                subroutine cart2mats(cart,N,ndim,nbeads,mats,nm_matrix)
                        integer, intent(in) :: N,ndim,nbeads
                        real(8), intent(in) :: cart(nbeads,ndim,N)
                        real(8), intent(inout) :: mats(nbeads,ndim,N)
                        real(8), intent(in) :: nm_matrix(nbeads,nbeads)

                        !This subroutine by default assumes that the first(i.e. along beads in fortran) 
                        !axis is the one to be transformed
                       
                        integer :: i,j
                        !OMP PARALLEL DO PRIVATE(i,j) SHARED(cart,mats)
                        do i=1,N
                                do j = 1,ndim
                                        mats(:,j,i) = matmul(nm_matrix,cart(:,j,i))
                                end do
                        end do
                        !OMP END PARALLEL DO
                end subroutine cart2mats

                subroutine mats2cart(mats,N,ndim,nbeads,cart,nm_matrix)
                        integer, intent(in) :: N,ndim,nbeads
                        real(8), intent(in) :: mats(nbeads,ndim,N)
                        real(8), intent(inout) :: cart(nbeads,ndim,N)
                        real(8), intent(in) :: nm_matrix(nbeads,nbeads)
                        real(8) :: nm_matrixt(nbeads,nbeads)
                        !This subroutine by default assumes that the first(i.e. along beads in fortran)
                        !axis is the one to be transformed

                        integer :: i,j
                        nm_matrixt(:,:) = transpose(nm_matrix)
                        !OMP PARALLEL DO PRIVATE(i,j) SHARED(cart,mats)
                        do i=1,N
                                do j = 1,ndim
                                        cart(:,j,i) = matmul(nm_matrixt,mats(:,j,i))
                                end do
                        end do
                        !OMP END PARALLEL DO
                end subroutine mats2cart

                subroutine cart2mats_hessian(cart,N,ndim,nbeads,mats,nm_matrix)
                        integer, intent(in) :: N,ndim,nbeads
                        real(8), intent(in) :: cart(nbeads,ndim,nbeads,ndim,N)
                        real(8), intent(inout) :: mats(nbeads,ndim,nbeads,ndim,N)
                        real(8), intent(in) :: nm_matrix(nbeads,nbeads)
                        real(8) :: nm_matrixt(nbeads,nbeads)

                        integer :: i,j,k
                        nm_matrixt(:,:) = transpose(nm_matrix)
                        !!OMP PARALLEL DO PRIVATE(i,j,k) SHARED(cart,mats)
                        do i=1,N
                                do j = 1,ndim
                                        do k = 1,ndim
                                                !mats(:,j,:,k,i) = matmul(nm_matrix,matmul(cart(:,j,:,k,i),nm_matrixt))
                                                mats(:,j,:,k,i) = &
                                                dgemm_matmul(nm_matrix,dgemm_matmul(cart(:,j,:,k,i),nm_matrixt,nbeads),nbeads)
                                        end do
                                end do
                        end do
                        !!OMP END PARALLEL DO
                end subroutine cart2mats_hessian

                subroutine mats2cart_hessian(mats,N,ndim,nbeads,cart,nm_matrix)
                        integer, intent(in) :: N,ndim,nbeads
                        real(8), intent(in) :: mats(nbeads,ndim,nbeads,ndim,N)
                        real(8), intent(inout) :: cart(nbeads,ndim,nbeads,ndim,N)
                        real(8), intent(in) :: nm_matrix(nbeads,nbeads)
                        real(8) :: nm_matrixt(nbeads,nbeads)

                        integer :: i,j,k
                        nm_matrixt(:,:) = transpose(nm_matrix)
                        !!OMP PARALLEL DO PRIVATE(i,j,k) SHARED(cart,mats)
                        do i=1,N
                                do j = 1,ndim
                                        do k = 1,ndim
                                                !cart(:,j,:,k,i) = matmul(nm_matrixt,matmul(mats(:,j,:,k,i),nm_matrix))
                                                cart(:,j,:,k,i) = &
                                                dgemm_matmul(nm_matrixt,dgemm_matmul(mats(:,j,:,k,i),nm_matrix,nbeads),nbeads)
                                        end do
                                end do
                        end do
                        !!OMP END PARALLEL DO
                end subroutine mats2cart_hessian
end module nmtrans


