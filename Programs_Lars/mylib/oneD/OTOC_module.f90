!compile using f2py -c OTOC_module.f90 -m fortran
module otoc_1d 
    use omp_lib
    implicit none
contains 

    subroutine test_openmpi()
        integer :: nthreads
        nthreads = -1
        nthreads = omp_get_num_threads()
        ! will print the number of running threads when compiled with OpenMP, else will print -1
        print*, "nthreads=", nthreads
        !print *, omp_get_max_threads()
    end subroutine test_openmpi

    subroutine which_version()
        print*,'Version:0'

    end subroutine which_version

    subroutine get_x_nm(X,EV,grid,N_grdpts,N_trunc) !N_trunc is amount of energy levels considered 
        integer, intent(in) :: N_grdpts
        integer, intent(in) :: N_trunc
        real(8), intent(in) :: EV(N_grdpts,N_trunc)
        real(8), intent(in) ::grid(N_grdpts)
        !f2py real(8), intent(in,out,copy) :: X(N_trunc,N_trunc)
        real(8), intent(out) :: X(N_trunc,N_trunc) !careful with type in python

        integer :: i
        integer :: n
        integer :: m

        X(:,:)=0.0

        do n=1, N_trunc
            do m=1, N_trunc
                do i=1,N_grdpts
                    X(n,m)= X(n,m)+EV(i,n)*EV(i,m)*grid(i)
                end do
            end do
        end do
    end subroutine get_x_nm

    subroutine get_b_nm(B,X, E, t, N_trunc, N_tsteps)
        integer, intent(in) :: N_trunc
        integer, intent(in) :: N_tsteps
        real(8), intent(in) :: X(N_trunc,N_trunc)
        real(8), intent(in) :: E(N_trunc)
        real(8), intent(in) :: t(N_tsteps)
        !f2py complex(8), intent(in,out,copy) :: B(N_trunc,N_trunc,N_tsteps)
        complex(8), intent(out) :: B(N_trunc,N_trunc,N_tsteps)
        complex(8), parameter :: i = (0.0, 1.0) 
        integer :: t_index,n,m,k
     
        B(:,:,:)=(0.0,0.0)
        
        do t_index= 1, size(t) !or simpy N_tsteps
            do m=1, N_trunc
                do n=1, N_trunc
                    do k=1, N_trunc
                        B(n,m,t_index)= B(n,m,t_index) + 0.5*X(n,k)*X(k,m)*&
                        &( (E(k)-E(m))*EXP(i*(E(n)-E(k))*t(t_index))&
                        & - (E(n)-E(k))*EXP(i*(E(k)-E(m))*t(t_index)) )
                    end do
                end do
            end do 
        end do
    end subroutine get_b_nm

    subroutine get_b_nm_omp(B,X, E, t, N_trunc, N_tsteps)
        integer, intent(in) :: N_trunc
        integer, intent(in) :: N_tsteps
        real(8), intent(in) :: X(N_trunc,N_trunc)
        real(8), intent(in) :: E(N_trunc)
        real(8), intent(in) :: t(N_tsteps)
        !f2py complex(8), intent(in,out,copy) :: B(N_trunc,N_trunc,N_tsteps)
        complex(8), intent(out) :: B(N_trunc,N_trunc,N_tsteps)
        complex(8), parameter :: i = (0.0, 1.0) 
        complex(8) :: sum = (0.0,0.0)
        integer :: t_index,n,m,k
        
     
        B(:,:,:)=(0.0,0.0)
        !$OMP PARALLEL DO default(none) SHARED(B) PRIVATE(t_index,m,n,k) 
        do t_index= 1, size(t) !or simpy N_tsteps
            do m=1, N_trunc
                do n=1, N_trunc
                    do k=1, N_trunc
                        !B(n,m,t_index)= B(n,m,t_index) + 0.5*X(n,k)*X(k,m)*&
                        !&( (E(k)-E(m))*EXP(i*(E(n)-E(k))*t(t_index))&
                        !& - (E(n)-E(k))*EXP(i*(E(k)-E(m))*t(t_index)) )
                        sum=0.5*X(n,k)*X(k,m)*&
                        &( (E(k)-E(m))*EXP(i*(E(n)-E(k))*t(t_index))&
                        & - (E(n)-E(k))*EXP(i*(E(k)-E(m))*t(t_index)) )
                        !$OMP critical
                        B(n,m,t_index)= B(n,m,t_index)+sum
                    end do
                end do
            end do 
        end do
        !$OMP END PARALLEL DO
    end subroutine get_b_nm_omp

    subroutine get_b_nm_dc(B,X, E, t, N_trunc, N_tsteps)
        integer, intent(in) :: N_trunc
        integer, intent(in) :: N_tsteps
        real(8), intent(in) :: X(N_trunc,N_trunc)
        real(8), intent(in) :: E(N_trunc)
        real(8), intent(in) :: t(N_tsteps)
        !f2py complex(8), intent(in,out,copy) :: B(N_trunc,N_trunc,N_tsteps)
        complex(8), intent(out) :: B(N_trunc,N_trunc,N_tsteps)
        complex(8), parameter :: i = (0.0, 1.0) 
        integer :: t_index,n,m,k
     
        B(:,:,:)=(0.0,0.0)
        
        do concurrent (t_index= 1: N_tsteps)
            do concurrent (m= 1: N_trunc)
                do concurrent (n= 1: N_trunc) 
                    do k=1, N_trunc
                        B(n,m,t_index)= B(n,m,t_index) + 0.5*X(n,k)*X(k,m)*&
                        &( (E(k)-E(m))*EXP(i*(E(n)-E(k))*t(t_index))&
                        & - (E(n)-E(k))*EXP(i*(E(k)-E(m))*t(t_index)) )
                    end do
                end do
            end do 
        end do
    end subroutine get_b_nm_DC

    subroutine get_c_n(C,B, N_trunc, N_tsteps)
        integer, intent(in) :: N_trunc
        integer, intent(in) :: N_tsteps
        complex(8), intent(in) :: B(N_trunc,N_trunc,N_tsteps)
        !f2py real(8), intent(in,out,copy) :: C(N_trunc,N_tsteps)
        real(8), intent(out) :: C(N_trunc, N_tsteps)
        integer :: t_index,n,m

        C(:,:)=0.0
        
        do t_index= 1, N_tsteps
            do n=1, N_trunc
                do m=1, N_trunc
                    C(n,t_index)= C(n,t_index) + ABS(B(n,m,t_index))**2
                end do
            end do 
        end do
    end subroutine get_c_n


end module otoc_1d


