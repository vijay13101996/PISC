!This module contains the subroutines required to compute the 'Krylov complexity' 
!of a system at finite temperature. Krylov complexity is a measure of the 'chaoticity'
!of the dynamics of a quantum system and is a quantity defined for a Hermitian operator. 

!Krylov complexity measures the temporal 'spread' of an operator in the available Hilbert 
!space due to the time evolution of the quantum Liouvillian. The 'Lanczos coefficients' are
!obtained upon orthonormalizing the canonical basis of this Hilbert space with respect to an 
!inner product. The growth of the Lanczos coefficients with the order of the expansion is a 
!measure of the 'chaoticity' of the system (and is equivalent to the Krylov complexity). 

!Apart from the Lanczos coefficients, the moments of the correlation function of the operator
!are also indicative of the chaoticity of the system. A similar measure of the 'chaoticity' can be
!obtained by computing the moments of the correlation function of the operator. Knowledge of the
!moments of the correlation function is equivalent to the knowledge of the Lanczos coefficients.

!In this module, we provide subroutines to compute the Lanczos coefficients of an operator and the
!moments of the correlation function of the operator. The subroutines are written in Fortran and 
!are as follows:

!1. pos_matrix_elts 
!   Computes the matrix elements of the position operator in the basis of the eigenvectors of the operator.

!2. compute_pos_matrix
!   Computes the matrix representation of the position operator in the basis of the eigenvectors of the operator.

!3. compute_mom_matrix
!   Computes the matrix representation of the momentum operator in the basis of the eigenvectors of the operator.

!4. compute_liouville_matrix
!   Computes the matrix representation of the Liouvillian in the basis of the eigenvectors of the operator.

!5. compute_ip
!   Computes the inner product of two operators as a trace of the product of the operators with the appropriate weight.
!   This is generally temperature-independent inner product.

!6. compute_hadamard_product
!   Computes the Hadamard product of two operators.

!7. compute_lanczos_coeffs
!   Subroutine to iteratively compute the Lanczos coefficients of an operator.

!8. compute_moments
!   Subroutine to compute the moments of the correlation function of an operator.

!Note that the f2py tags are used to call the Fortran subroutines from Python. The same code can be used for computing
!position matrix elements in 1D and 2D by commenting out the appropriate lines in the pos_matrix_elts subroutine.

!The command to generate the python module from this Fortran code is:
!f2py3 -c -m --f90flags="-O3" Krylov_complexity Krylov_complexity.f90

module krylov_complexity
        use omp_lib
        implicit none
        private
        public :: pos_matrix_elts
        public :: compute_pos_matrix
        public :: compute_mom_matrix
        public :: compute_liouville_matrix
        public :: compute_ip
        public :: compute_hadamard_product
        public :: compute_lanczos_coeffs
        public :: compute_On_matrix
        public :: compute_moments

        contains
                subroutine pos_matrix_elts(vecs,len1vecs,len2vecs,x_arr,lenx,dx,dy,n,k,pos_mat_elt)
                        !Computes the matrix elements of the position operator in the basis of the eigenvectors of the operator.
                        !vecs: Eigenvectors of the operator
                        !len1vecs: Number of rows of vecs
                        !len2vecs: Number of columns of vecs
                        !x_arr: Array of x values
                        !lenx: Number of x values
                        !dx: Spacing between x values
                        !dy: Spacing between y values
                        !n,k: Indices of the matrix element
                        !pos_mat_elt: Matrix element of the position operator


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
                        !Computes the matrix representation of the position operator in the basis of the eigenvectors of the operator.
                        !vecs: Eigenvectors of the operator
                        !len1vecs: Number of rows of vecs
                        !len2vecs: Number of columns of vecs
                        !x_arr: Array of x values
                        !lenx: Number of x values
                        !dx: Spacing between x values
                        !dy: Spacing between y values
                        !pos_mat: Matrix representation of the position operator


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
                        !Computes the matrix representation of the Liouvillian in the basis of the eigenvectors of the operator.
                        !vals: Eigenvalues of the operator
                        !lenvals: Number of eigenvalues
                        !liou_mat: Matrix representation of the Liouvillian

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
                        !Computes the matrix representation of the momentum operator in the basis of the eigenvectors of the operator.
                        !vecs: Eigenvectors of the operator
                        !len1vecs: Number of rows of vecs
                        !len2vecs: Number of columns of vecs
                        !vals: Eigenvalues of the operator
                        !x_arr: Array of x values
                        !lenx: Number of x values
                        !dx: Spacing between x values
                        !dy: Spacing between y values
                        !m: Mass of the particle

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

                subroutine compute_ip(O1,O2,lenO,beta,vals,lamda,ip,ipkey)
                        !Computes all forms of inner products of two operators.
                        !O1,O2: Operators
                        !lenO: Dimension of the operators
                        !beta: Inverse temperature
                        !vals: Eigenvalues of the operator
                        !lamda: Number between 0 to 1 to specify the asymmetry of the inner product
                        !ip: Inner product
                        !ipkey: Key to specify the type of inner product

                        integer, intent(in) :: lenO
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O1
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O2
                        real(kind=8), intent(in) :: beta
                        real(kind=8), dimension(lenO), intent(in) :: vals
                        real(kind=8), intent(in) :: lamda
                        character(len=3), intent(in) :: ipkey

                        !f2py real(kind=8), intent(in,out,copy) :: ip
                        real(kind=8), intent(inout) :: ip
                        
                        real(kind=8) :: Z, wgt
                        integer :: i,j
                        
                        ip = 0.0d0
                        Z = 0.0d0
                        wgt = 0.0d0

                        do i=1,lenO
                                do j=1,lenO
                                        if (ipkey .eq. 'dir') then
                                                ip = ip + conjg(O1(j,i))*O2(j,i)
                                        else if (ipkey .eq. 'wgm') then
                                                ip = ip + exp(-0.5*beta*(vals(i)+vals(j)))*conjg(O1(j,i))*O2(j,i)
                                        else if (ipkey .eq. 'std') then
                                                ip = ip + exp(-beta*vals(i))*conjg(O1(j,i))*O2(j,i)
                                        else if (ipkey .eq. 'asm') then
                                                wgt = exp(-beta*lamda*vals(i))*exp(-beta*(1.0d0-lamda)*vals(j))
                                                ip = ip + wgt*conjg(O1(j,i))*O2(j,i)
                                        else if (ipkey .eq. 'kbo') then
                                                if (i .ne. j) then !Take care of degeneracies when required!
                                                        wgt = (exp(-beta*vals(i)) - exp(-beta*vals(j)))/(beta*(vals(j)-vals(i)))
                                                        ip = ip + wgt*conjg(O1(j,i))*O2(j,i)
                                                else
                                                        ip = ip + exp(-beta*vals(i))*conjg(O1(j,i))*O2(j,i)
                                        
                                                end if
                                        end if
                                end do
                                !print*, 'Zinloop',Z
                                Z = Z + exp(-beta*vals(i)) 
                        end do
                        ip = ip/Z

                end subroutine compute_ip

                subroutine compute_hadamard_product(O1,O2,lenO,prod)
                        !Computes the Hadamard product of two operators.
                        !O1,O2: Operators
                        !lenO: Dimension of the operators
                        !prod: Hadamard product of the operators

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

                subroutine compute_lanczos_coeffs(O, L, lenO, barr, ncoeffs, beta, vals, lamda, ipkey)
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
                        real(kind=8), intent(in) :: lamda
                        character(len=3), intent(in) :: ipkey
                        
                        !f2py real(kind=8), dimension(ncoeffs), intent(in,out,copy) :: barr
                        real(kind=8), dimension(ncoeffs), intent(inout) :: barr
                        
                        integer :: n
                        complex(kind=8), dimension(lenO,lenO) :: On1, On2, Otemp, An
                        real(kind=8) :: tempvar

                        call compute_ip(O,O,lenO,beta,vals,lamda,tempvar,ipkey)
                        barr(1) = sqrt(tempvar)
                        On2 = O/barr(1)
                        
                        call compute_hadamard_product(L,On2,lenO,An)

                        call compute_ip(An,An,lenO,beta,vals,lamda,tempvar,ipkey)
                        barr(2) = sqrt(tempvar)
                        On1 = An/barr(2)

                        do n=3,ncoeffs
                                if (barr(n-1) < 1.0d-8) then
                                        print *, 'barr(n) = 0.0d0'
                                        exit
                                end if

                                call compute_hadamard_product(L,On1,lenO,Otemp)
                                An = Otemp - barr(n-1)*On2

                                call compute_ip(An,An,lenO,beta,vals,lamda,tempvar,ipkey)
                                barr(n) = sqrt(tempvar)
                                On2 = On1
                                On1 = An/barr(n)
                        end do
 
                end subroutine compute_lanczos_coeffs

                subroutine compute_On_matrix(O, L, lenO, barr, ncoeffs, beta, vals, lamda, ipkey, On, nmat)
                        !Compute the coefficients of the Lanczos expansion of the operator O
                        !O: lenO x lenO matrix
                        !ncoeffs : number of coefficients to compute
                        
                        !Step 1: Set O_{-1} = 0, define L matrix.
                        !Step 2: Set O0 = O/||O||
                        !Step 3: For i = 1 to ncoeffs, compute Oi = L*Oi-1 - bi-1*Oi-2
                        !Step 4: Compute bi = ||Oi||, normalize Oi to get Oi = Oi/bi

                        integer, intent(in) :: lenO, ncoeffs, nmat
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: L
                        real(kind=8), dimension(lenO), intent(in) :: vals
                        real(kind=8), intent(in) :: beta
                        real(kind=8), intent(in) :: lamda
                        character(len=3), intent(in) :: ipkey
                        
                        !f2py real(kind=8), dimension(ncoeffs), intent(in,out,copy) :: barr
                        real(kind=8), dimension(ncoeffs), intent(inout) :: barr

                        !f2py complex(kind=8), dimension(lenO,lenO), intent(in,out,copy) :: On
                        complex(kind=8), dimension(lenO,lenO), intent(inout) :: On
                        
                        integer :: n
                        complex(kind=8), dimension(lenO,lenO) :: On1, On2, Otemp, An
                        real(kind=8) :: tempvar

                        call compute_ip(O,O,lenO,beta,vals,lamda,tempvar,ipkey)
                        barr(1) = sqrt(tempvar)
                        if (nmat .eq. 1) then
                                On = O
                                !Exit subroutine
                                return
                        end if
                        On2 = O/barr(1)

                        call compute_hadamard_product(L,On2,lenO,An)
                        call compute_ip(An,An,lenO,beta,vals,lamda,tempvar,ipkey)
                        barr(2) = sqrt(tempvar)

                        if (nmat .eq. 2) then
                                On = An
                                !Exit subroutine
                                return
                        end if
                        On1 = An/barr(2)
                        
                        do n=3,ncoeffs
                                if (barr(n-1) < 1.0d-8) then
                                        print *, 'barr(n) = 0.0d0'
                                        exit
                                end if

                                call compute_hadamard_product(L,On1,lenO,Otemp)
                                An = Otemp - barr(n-1)*On2
                                
                                call compute_ip(An,An,lenO,beta,vals,lamda,tempvar,ipkey)
                                barr(n) = sqrt(tempvar)
                                if (n .eq. nmat) then
                                        On = An
                                        !Break the loop
                                        return
                                end if
                                On2 = On1
                                On1 = An/barr(n)
                        end do
 
                end subroutine compute_On_matrix

                subroutine compute_moments(O,lenO,vals,beta,ipkey,lamda,moments,nmoments)
                        !Compute the moments of the correlation function of the operator O
                        !O: lenO x lenO matrix
                        !lenO: Dimension of the operator
                        !vals: Eigenvalues of the operator
                        !nmoments: number of moments to compute
                        !moments: array to store the moments
                        !beta: Inverse temperature

                        integer, intent(in) :: lenO,nmoments
                        complex(kind=8), dimension(lenO,lenO), intent(in) :: O
                        real(kind=8), dimension(lenO), intent(in) :: vals
                        real(kind=8), intent(in) :: beta
                        character(len=3), intent(in) :: ipkey
                        real(kind=8), intent(in) :: lamda
                                
                        !f2py real(kind=8), dimension(nmoments), intent(in,out,copy) :: moments
                        real(kind=8), dimension(nmoments), intent(inout) :: moments

                        real(kind=8) :: tempvar, Z, pref, wgt
                        integer :: i,j,n
        
                        
                        Z = 0.0d0
                        do i=1,lenO
                                Z = Z + exp(-beta*vals(i))
                        end do

                        moments = 0.0d0

                        do n=1,nmoments
                                do i=1,lenO
                                        do j=1,lenO
                                        pref = (vals(i) - vals(j))**(n-1)*conjg(O(j,i))*O(j,i)
                                        if (ipkey .eq. 'dir') then
                                                wgt = 1.0d0
                                        else if (ipkey .eq. 'wgm') then
                                                wgt = exp(-0.5*beta*(vals(i)+vals(j)))
                                        else if (ipkey .eq. 'std') then
                                                wgt = exp(-beta*vals(i))
                                        else if (ipkey .eq. 'asm') then
                                                wgt = exp(-beta*lamda*vals(i))*exp(-beta*(1.0d0-lamda)*vals(j))
                                        else if (ipkey .eq. 'kbo') then
                                                if (i .ne. j) then !Take care of degeneracies when required!
                                                        wgt = (exp(-beta*vals(i)) - exp(-beta*vals(j)))/(beta*(vals(j)-vals(i)))
                                                else
                                                        wgt = exp(-beta*vals(i)) 
                                                end if
                                        end if
                                        moments(n) = moments(n) + wgt*pref
                                        end do
                                        !        moments(n) = moments(n) + (vals(i) - vals(j))**(n-1) &
                                        !        *exp(-0.5*beta*(vals(i)+vals(j)))*conjg(O(j,i))*O(j,i)
                                        !end do
                                end do
                                moments(n) = moments(n)/Z
                        end do 

                end subroutine compute_moments 

end module krylov_complexity







