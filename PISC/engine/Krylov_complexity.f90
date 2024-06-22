module krylov_complexity
        use omp_lib
        implicit none
        private
        public :: pos_matrix_elts
        public :: compute_pos_matrix
        public :: compute_mom_matrix
        public :: compute_liouville_matrix
        public :: compute_direct_ip
        public :: compute_wightman_ip
        public :: compute_hadamard_product
        public :: compute_lanczos_coeffs

        contains
                subroutine pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k,pos_mat_elt)
                        integer, intent(in) :: n,k,len1vecs,len2vecs,lenx
                        real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
                        real(kind=8), dimension(lenx), intent(in) :: x_arr
                        real(kind=8), intent(in) ::  dx,dy
                        !f2py real(kind=8), intent(in,out,copy) :: pos_mat_elt
                        integer :: i,j
                        real(kind=8), intent(inout) :: pos_mat_elt
                        pos_mat_elt = 0.0d0
                        do i = 1,len1vecs
                                pos_mat_elt = pos_mat_elt + vecs(i,n)*vecs(i,k)*x_arr(i)*dx!*dy !! Change here for 1D 
                        end do
                end subroutine pos_matrix_elts

                subroutine compute_pos_matrix(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,pos_mat)
                        integer, intent(in) :: len1vecs,len2vecs,lenx
                        real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
                        real(kind=8), dimension(lenx), intent(in) :: x_arr
                        real(kind=8), intent(in) ::  dx,dy
                        real(kind=8), dimension(len2vecs,len2vecs) :: pos_mat
                        integer :: i,j
                        do i = 1,len2vecs
                                do j = 1,len2vecs
                                        call pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,i,j,pos_mat(i,j))
                                end do
                        end do
                end subroutine compute_pos_matrix

                subroutine compute_liouville_matrix(vals,lenvals,liou_mat)
                        integer, intent(in) :: lenvals
                        real(kind=8), dimension(lenvals), intent(in) :: vals
                        real(kind=8), dimension(lenvals,lenvals) :: liou_mat
                        integer :: i,j
                        do i = 1,lenvals
                                do j = 1,lenvals
                                        liou_mat(i,j) = vals(i) - vals(j)
                                end do
                        end do
                end subroutine compute_liouville_matrix

                subroutine compute_mom_matrix(vecs,len1vecs,len2vecs,vals,x_arr,m,lenx,dx,dy,mom_mat)
                        integer, intent(in) :: len1vecs,len2vecs,lenx
                        real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
                        real(kind=8), dimension(len2vecs), intent(in) :: vals
                        real(kind=8), dimension(lenx), intent(in) :: x_arr
                        real(kind=8), intent(in) ::  dx,dy,m
                        real(kind=8), dimension(len2vecs,len2vecs) :: mom_mat
                        real(kind=8) :: Eij
                        integer :: i,j
                        do i = 1,len2vecs
                                do j = 1,len2vecs
                                        call pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,i,j,mom_mat(i,j))
                                        Eij = vals(i) - vals(j)
                                        mom_mat(i,j) = cmplx(0.0d0,1.0d0)*mom_mat(i,j)*Eij*m
                                end do
                        end do
                end subroutine compute_mom_matrix

                subroutine compute_direct_ip(O1,O2,lenO,ip)
                        integer, intent(in) :: lenO
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O1
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O2
                        real(kind=8), intent(out) :: ip
                        integer :: i
                        ip = 0.0d0
                        do i = 1,lenO
                                ip = ip + conjg(O1(i,i))*O2(i,i)
                        end do
                end subroutine compute_direct_ip

                subroutine compute_wightman_ip(O1,O2,lenO,beta,vals,ip)
                        integer, intent(in) :: lenO
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O1
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O2
                        real(kind=8), intent(in) :: beta
                        real(kind=8), dimension(lenO), intent(in) :: vals
                        real(kind=8), intent(out) :: ip
                        integer :: i,j
                        ip = 0.0d0
                        do i=1,lenO
                                do j=1,lenO
                                        ip = ip + exp(-0.5*beta*(vals(i)+vals(j)))*conjg(O1(i,j))*O2(j,i)
                                end do
                        end do
                end subroutine compute_wightman_ip

                subroutine compute_hadamard_product(O1,O2,lenO,prod)
                        integer, intent(in) :: lenO
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O1
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O2
                        complex(kind=8), dimension(lenO,lenO), intent(out) :: prod
                        integer :: i,j
                        do i=1,lenO
                                do j=1,lenO
                                        prod(i,j) = O1(i,j)*O2(i,j)
                                end do
                        end do
                end subroutine compute_hadamard_product

                subroutine compute_lanczos_coeffs(O,L, lenO, ncoeffs, ip)

                        !Compute the coefficients of the Lanczos expansion of the operator O
                        !O: lenO x lenO matrix
                        !ncoeffs : number of coefficients to compute
                        
                        !Step 1: Set b0 = 0, O_{-1} = 0, define L matrix.
                        !Step 2: Set O0 = O/||O||
                        !Step 3: For i = 1 to ncoeffs, comput Oi = L*Oi-1 - bi-1*Oi-2
                        !Step 4: Compute bi = ||Oi||, normalize Oi to get Oi = Oi/bi


                        integer, intent(in) :: lenO, ncoeffs
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O
                        real(kind=8), dimension(lenO,lenO), intent(in) :: L
                        character(len=3), intent(in) :: ip
                        integer :: n
                        complex(kind=8), dimension(lenO,lenO) :: On1, On2, Otemp
                        real(kind=8), dimension(ncoeffs) :: barr
                        real(kind=8) :: tempvar
                        
                        On1 = O
                        On2 = O

                        do n=1, ncoeffs
                                if (n == 1) then
                                        !Step 1
                                        call compute_direct_ip(O, O, lenO, tempvar)
                                        barr(n) = sqrt(tempvar)
                                else
                                        !Step 1
                                        call compute_hadamard_product(L, On1, lenO, Otemp)
                                        Otemp = Otemp - barr(n-1)*On2
                                        
                
                                        call compute_direct_ip(O,O0, lenO, bi)
                                        call compute_direct_ip(O0,O0, lenO, bim1)
                                        call compute_hadamard_product(O,bi, lenO, Oi)
                                        call compute_hadamard_product(O0,bim1, lenO, Oim1)
                                        call compute_hadamard_product(Oi,1.0d0/bi, lenO, Oi)
                                end if
                        end do
 
                end subroutine compute_lanczos_coeffs

end module krylov_complexity







