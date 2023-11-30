
module integrator
	use omp_lib
        implicit none
	private
	public :: A_f
	public :: B_f
	public :: M_upd_f
        public :: thalfstep
        public :: A_f_ndt
        public :: pq_step_f

	contains 
		subroutine A_f(q,p,N,ndim,nbeads,qdt,dynm3)   
                        ! Implements the A step of a symplectic integrator
                        
                        ! q: Position in Matsubara(normal mode) coordinates
                        ! p: Momentum in Matsubara(normal mode) coordinates
                        ! N: Number of particles
                        ! ndim: Number of dimensions
                        ! nbeads: Number of beads
                        ! qdt: Time step for position
                        ! dynm3: Dynamical mass matrix (used for CMD/PA-CMD simulations)

                        ! Note: This step remains the same for all orders of symplectic integrators

                        integer, intent(in) :: N,ndim,nbeads 
                        real(8), intent(in) :: qdt
                        real(8), dimension(nbeads,ndim,N), intent(inout) :: q
                        real(8), dimension(nbeads,ndim,N), intent(in) :: p
                        real(8), dimension(nbeads,ndim,N), intent(in) :: dynm3
                        !Arrays are passed in reversed order to make them fortran contiguous

                        integer :: i,j,k

			!!$OMP PARALLEL DO PRIVATE(i,j,k) SHARED(q,p,dynm3)
                        do i=1,N
                                do k=1,nbeads
                                        do j=1,ndim
                                                q(k,j,i)=q(k,j,i)+qdt*p(k,j,i)/dynm3(k,j,i)
                                        end do
                                end do
                        end do
                        !!$OMP END PARALLEL DO
		end subroutine A_f
			
		subroutine B_f(p,dpot,N,ndim,nbeads,pdt,centmove,nmats)     
                        ! Implements the B step of a symplectic integrator

                        ! p: Momentum in Matsubara(normal mode) coordinates
                        ! dpot: Derivative of potential energy (in Matsubara(normal mode) coordinates)
                        ! N: Number of particles
                        ! ndim: Number of dimensions
                        ! nbeads: Number of beads
                        ! pdt: Time step for momentum
                        ! centmove: If true, then the centroid is propagated in time
                        ! nmats: Number of normal modes to be propagated in time
                        
                        !Notes:
                        !1. The same function is used for both spring forces and potential forces.
                        !2. If nmats<nbeads, then all but the first nmats modes are propagated in time
                        !   (The above is used only for mean-field Matsubara dynamics)
                        !3. For normal RPMD simulations, nmats=nbeads

                        integer, intent(in) :: N,ndim,nbeads,nmats 
                        real(8), dimension(nbeads,ndim,N), intent(inout) :: p
                        real(8), dimension(nbeads,ndim,N), intent(in) :: dpot
                        !Arrays are passed in reversed order to make them fortran contiguous
                        
                        real(8), intent(in) :: pdt
                        integer :: i,j,k
                        logical, intent(in) :: centmove
                                     
                        !!$OMP PARALLEL DO PRIVATE(i,j,k) SHARED(p,dpot)
                        do i=1,N
                                do k=1,nbeads
                                        do j=1,ndim
                                                if (centmove .eqv. .true. .and. nmats==nbeads) then
                                                        p(k,j,i)=p(k,j,i)-pdt*dpot(k,j,i)
                                                else if (centmove .eqv. .false. .and. nmats==nbeads) then
                                                        if (k .NE. 1) then
                                                                p(k,j,i)=p(k,j,i)-pdt*dpot(k,j,i)
                                                        end if
                                                else
                                                        if (k .ge. nmats) then !This needs to be checked
                                                                p(k,j,i)=p(k,j,i)-pdt*dpot(k,j,i)
                                                        end if
                                                end if
                                        end do
                                end do
                        end do
                        !!$OMP END PARALLEL DO
		end subroutine B_f
	
		subroutine M_upd_f(Ma,Mb,dynm3,N,ndim,nbeads,qdt)
                        ! Updates the Monodromy matrix Mqp and Mqq (Hessian is not involved in this update step)
                        ! Ma: Monodromy matrix to be changed
                        ! Mb: Monodromy matrix to be used for updating Ma
                        ! dynm3: Dynamical mass matrix (used for CMD/PA-CMD simulations)
                        ! N: Number of particles
                        ! ndim: Number of dimensions
                        ! nbeads: Number of beads
                        ! qdt: Time step for Monodromy matrix update (same as qdt in A_f)

                        integer, intent(in) :: N,ndim,nbeads
                        real(8), dimension(nbeads,ndim,nbeads,ndim,N), intent(inout) :: Ma
                        real(8), dimension(nbeads,ndim,nbeads,ndim,N), intent(in) :: Mb
                        real(8), dimension(nbeads,ndim,N), intent(in) :: dynm3
                        !Arrays are passed in reversed order to make them fortran contiguous

                        real(8), intent(in) :: qdt
                        integer :: i,j,k,l,m

                        !Note that the array ordering is different from the order of 
                        !the loops. This is to improve performance: Fortran loops are
                        !faster when the outermost loop is the fastest varying index.

                        !!OMP PARALLEL DO PRIVATE(i,j,k,l,m) SHARED(Ma,Mb,dynm3)
                        do i=1,N
                                do l=1,nbeads
                                        do j=1,nbeads
                                                do m=1,ndim
                                                        do k=1,ndim
                                                                Ma(l,m,j,k,i)=Ma(l,m,j,k,i)+qdt*Mb(l,m,j,k,i)/dynm3(l,m,i)
                                                        end do
                                                end do
                                        end do
                                end do 
                        end do
                        !!OMP END PARALLEL DO
                end subroutine M_upd_f

		subroutine thalfstep(p,sm3,N,ndim,nbeads,rng_nm,c1,c2,pc,nmats)
                        ! Implements the half step of the thermostat
                        ! p: Momentum in Matsubara(normal mode) coordinates
                        ! sm3: Square root of the mass matrix (in Matsubara(normal mode) coordinates)
                        ! N: Number of particles
                        ! ndim: Number of dimensions
                        ! nbeads: Number of beads
                        ! rng_nm: Random number array (in Matsubara(normal mode) coordinates)
                        ! c1: Coefficient C1 of the thermostat
                        ! c2: Coefficient C2 of the thermostat

                        integer, intent(in) :: N,ndim,nbeads,nmats
                        real(8), dimension(nbeads,ndim,N), intent(inout) :: p
                        real(8), dimension(nbeads,ndim,N), intent(in) :: sm3
                        real(8), dimension(nbeads,ndim,N), intent(in) :: rng_nm
                        !Arrays are passed in reversed order to make them fortran contiguous
                        
                        real(8), dimension(nbeads), intent(in) :: c1,c2
                        logical, intent(in) :: pc
                        real(8), dimension(nbeads,ndim,N) :: sp
                        integer :: i,j,k
                        sp = 0.0d0

                        !!OMP PARALLEL DO PRIVATE(i,j,k) SHARED(p,sm3,rng_nm,sp,c1,c2)
                        do i=1,N
                                do k=1,nbeads 
                                        do j=1,ndim
                                                sp(k,j,i)=p(k,j,i)/sm3(k,j,i) 
                                                sp(k,j,i)=sp(k,j,i)*c1(k) + c2(k)*rng_nm(k,j,i)
                                                if (pc .eqv. .true. .and. nmats==nbeads) then
                                                        p(k,j,i)=sp(k,j,i)*sm3(k,j,i)
                                                else if (pc .eqv. .false. .and. nmats==nbeads) then
                                                        if (k .NE. 1) then
                                                                p(k,j,i)=sp(k,j,i)*sm3(k,j,i)
                                                        end if
                                                else
                                                        if (k .ge. nmats) then !This needs to be checked
                                                                p(k,j,i)=sp(k,j,i)*sm3(k,j,i)
                                                        end if
                                                end if
                                        end do
                                end do
                        end do
                        !!OMP END PARALLEL DO
		end subroutine thalfstep

                subroutine A_f_ndt(q,p,N,ndim,nbeads,qdt,ndt,dynm3)
                        ! Same as A_f, but the step is repeated ndt times
                        ! (Used primarily to compare loop speeds in fortran/python, when
                        !  the function calls inside the loops are implemented in fortran)

                        integer, intent(in) :: N,ndim,nbeads,ndt
                        real(8), intent(in) :: qdt
                        real(8), dimension(nbeads,ndim,N), intent(inout) :: q
                        real(8), dimension(nbeads,ndim,N), intent(in) :: p
                        real(8), dimension(nbeads,ndim,N), intent(in) :: dynm3
                        
                        integer :: i
                        do i=1,ndt
                                call A_f(q,p,N,ndim,nbeads,qdt,dynm3)
                        end do
                end subroutine A_f_ndt

                subroutine pq_step_f(q,p,N,ndim,nbeads,qdt,pdt,dynm3)
                        ! Implements one full velocity verlet step of a symplectic integrator
                        ! (Currently incomplete, since force is not updated after the A step)

                        integer, intent(in) :: N,ndim,nbeads
                        real(8), intent(in) :: qdt,pdt
                        real(8), dimension(nbeads,ndim,N), intent(inout) :: q
                        real(8), dimension(nbeads,ndim,N), intent(inout) :: p
                        real(8), dimension(nbeads,ndim,N), intent(in) :: dynm3

                        call A_f(q,p,N,ndim,nbeads,qdt,dynm3)
                        call B_f(p,dynm3,N,ndim,nbeads,pdt,.true.,nbeads)
                        call A_f(q,p,N,ndim,nbeads,qdt,dynm3)
                end subroutine pq_step_f
                        

end module integrator
