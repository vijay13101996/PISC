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
        public :: compute_moments
        contains
                subroutine pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k,pos_mat_elt)
                        integer, intent(in) :: n,k,len1vecs,len2vecs,lenx
                        real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
                        real(kind=8), dimension(lenx), intent(in) :: x_arr
                        real(kind=8), intent(in) ::  dx,dy
                        
                        !f2py real(kind=8), intent(in,out,copy) :: pos_mat_elt
                        real(kind=8), intent(inout) :: pos_mat_elt
                        
                        integer :: i,j
                        
                        pos_mat_elt = 0.0d0
                        do i = 1,len1vecs
                                !print *, vecs(i,n),vecs(i,k),x_arr(i)
                                pos_mat_elt = pos_mat_elt + vecs(i,n)*vecs(i,k)*x_arr(i)*dx*dy !! Change here for 1D 
                        end do
                end subroutine pos_matrix_elts

                subroutine compute_pos_matrix(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,pos_mat)
                        integer, intent(in) :: len1vecs,len2vecs,lenx
                        real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
                        real(kind=8), dimension(lenx), intent(in) :: x_arr
                        real(kind=8), intent(in) ::  dx,dy

                        !f2py real(kind=8), dimension(len2vecs,len2vecs), intent(in,out,copy) :: pos_mat
                        real(kind=8), dimension(len2vecs,len2vecs), intent(inout) :: pos_mat
                        
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
                        
                        !f2py real(kind=8), dimension(lenvals,lenvals), intent(in,out,copy) :: liou_mat
                        real(kind=8), dimension(lenvals,lenvals), intent(inout) :: liou_mat
                        
                        integer :: i,j
        
                        do i = 1,lenvals
                                do j = 1,lenvals
                                        liou_mat(i,j) = (vals(i) - vals(j))
                                end do
                        end do
                end subroutine compute_liouville_matrix

                subroutine compute_mom_matrix(vecs,len1vecs,len2vecs,vals,x_arr,m,lenx,dx,dy,mom_mat)
                        integer, intent(in) :: len1vecs,len2vecs,lenx
                        real(kind=8), dimension(len1vecs,len2vecs), intent(in) :: vecs
                        real(kind=8), dimension(len2vecs), intent(in) :: vals
                        real(kind=8), dimension(lenx), intent(in) :: x_arr
                        real(kind=8), intent(in) ::  dx,dy,m
                        
                        !f2py complex(kind=8), dimension(len2vecs,len2vecs), intent(in,out,copy) :: mom_mat
                        complex(kind=8), dimension(len2vecs,len2vecs), intent(inout) :: mom_mat
                        real(kind=8) :: Eij, pos_mat_elt
                        integer :: i,j
                        do i = 1,len2vecs
                                do j = 1,len2vecs
                                        call pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,i,j,pos_mat_elt)
                                        Eij = vals(i) - vals(j)
                                        mom_mat(i,j) = cmplx(0.0d0,1.0d0)*Eij*m*pos_mat_elt
                                end do
                        end do
                end subroutine compute_mom_matrix

                subroutine compute_direct_ip(O1,O2,lenO,ip)
                        integer, intent(in) :: lenO
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O1
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O2
                        
                        !f2py real(kind=8), intent(in,out,copy) :: ip
                        real(kind=8), intent(out) :: ip

                        integer :: i,j
                        ip = 0.0d0
                        do i = 1,lenO
                                do j = 1,lenO
                                        ip = ip + conjg(O1(j,i))*O2(j,i)
                                end do
                        end do
                end subroutine compute_direct_ip

                subroutine compute_wightman_ip(O1,O2,lenO,beta,vals,ip)
                        integer, intent(in) :: lenO
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O1
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O2
                        real(kind=8), intent(in) :: beta
                        real(kind=8), dimension(lenO), intent(in) :: vals
                        
                        !f2py real(kind=8), intent(in,out,copy) :: ip
                        real(kind=8), intent(inout) :: ip

                        integer :: i,j
                        real(kind=8) :: Z

                        ip = 0.0d0
                        Z = 0.0d0

                        do i=1,lenO
                                do j=1,lenO
                                        ip = ip + exp(-0.5*beta*(vals(i)+vals(j)))*conjg(O1(j,i))*O2(j,i)
                                end do
                                Z = Z + exp(-beta*vals(i))
                        end do
                        ip = ip/Z
                end subroutine compute_wightman_ip

                subroutine compute_hadamard_product(O1,O2,lenO,prod)
                        integer, intent(in) :: lenO
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O1
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O2

                        !f2py complex(kind=8), dimension(lenO,lenO), intent(in,out,copy) :: prod
                        complex(kind=8), dimension(lenO,lenO), intent(inout) :: prod

                        integer :: i,j
                        do i=1,lenO
                                do j=1,lenO
                                        prod(i,j) = O1(i,j)*O2(i,j)
                                end do
                        end do
                end subroutine compute_hadamard_product

                subroutine compute_lanczos_coeffs(O, L, lenO, barr, ncoeffs, beta, vals, ipkey)

                        !Compute the coefficients of the Lanczos expansion of the operator O
                        !O: lenO x lenO matrix
                        !ncoeffs : number of coefficients to compute
                        
                        !Step 1: Set O_{-1} = 0, define L matrix.
                        !Step 2: Set O0 = O/||O||
                        !Step 3: For i = 1 to ncoeffs, comput Oi = L*Oi-1 - bi-1*Oi-2
                        !Step 4: Compute bi = ||Oi||, normalize Oi to get Oi = Oi/bi

                        integer, intent(in) :: lenO, ncoeffs
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: L
                        real(kind=8), dimension(lenO), intent(in) :: vals
                        real(kind=8), intent(in) :: beta
                        character(len=3), intent(in) :: ipkey
                        
                        !f2py real(kind=8), dimension(ncoeffs), intent(in,out,copy) :: barr
                        real(kind=8), dimension(ncoeffs), intent(inout) :: barr
                        
                        integer :: n
                        complex(kind=8), dimension(lenO,lenO) :: On1, On2, Otemp, An
                        real(kind=8) :: tempvar

                        if (ipkey .eq. 'dir') then
                                call compute_direct_ip(O,O,lenO,tempvar)
                        else if (ipkey .eq. 'wgm') then
                                call compute_wightman_ip(O,O,lenO,beta,vals,tempvar)
                        end if
                        barr(1) = sqrt(tempvar)
                        On2 = O/barr(1)
                        
                        call compute_hadamard_product(L,On2,lenO,An)
                        if (ipkey .eq. 'dir') then
                                call compute_direct_ip(An,An,lenO,tempvar)
                        else if (ipkey .eq. 'wgm') then
                                call compute_wightman_ip(An,An,lenO,beta,vals,tempvar)
                        end if
                        barr(2) = sqrt(tempvar)
                        On1 = An/barr(2)

                        do n=3,ncoeffs
                                if (barr(n-1) < 1.0d-5) then
                                        print *, 'barr(n) = 0.0d0'
                                        exit
                                end if

                                call compute_hadamard_product(L,On1,lenO,Otemp)
                                An = Otemp - barr(n-1)*On2
                                if (ipkey .eq. 'dir') then
                                        call compute_direct_ip(An,An,lenO,tempvar)
                                else if (ipkey .eq. 'wgm') then
                                        call compute_wightman_ip(An,An,lenO,beta,vals,tempvar)
                                end if
                                barr(n) = sqrt(tempvar)
                                On2 = On1
                                On1 = An/barr(n)
                        end do
 
                end subroutine compute_lanczos_coeffs

                subroutine compute_moments(O,vals,lenO,beta,nmoments,moments)
                        integer, intent(in) :: lenO,nmoments
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O
                        real(kind=8), dimension(lenO), intent(in) :: vals
                        real(kind=8), intent(in) :: beta

                        !f2py real(kind=8), dimension(nmoments), intent(in,out,copy) :: moments
                        real(kind=8), dimension(nmoments), intent(inout) :: moments

                        real(kind=8) :: tempvar, Z
                        integer :: i,j,n

                        Z = 0.0d0
                        do i=1,lenO
                                Z = Z + exp(-beta*vals(i))
                        end do

                        moments = 0.0d0

                        do n=1,nmoments
                                do i=1,lenO
                                        do j=1,lenO
                                                moments(n) = moments(n) + (vals(i) - vals(j))**(n-1) &
                                                *exp(-0.5*beta*(vals(i)+vals(j)))*conjg(O(j,i))*O(j,i)
                                        end do
                                end do
                                moments(n) = moments(n)/Z
                        end do 

                end subroutine compute_moments 

end module krylov_complexity







