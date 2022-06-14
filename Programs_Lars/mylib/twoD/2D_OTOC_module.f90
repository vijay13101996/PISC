!compile using f2py -c OTOC_module.f90 -m fortran
module otoc_2d 
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
    
    !Bottleneck
    subroutine get_b_omp(B,X, E, t, N_trunc, N_tsteps)!add Y if not correct
        integer, intent(in) :: N_trunc
        integer, intent(in) :: N_tsteps
        real(8), intent(in) :: X(N_trunc,N_trunc)
        !real(8), intent(in) :: Y(N_trunc,N_trunc)
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
                        !sum=0.5*(0.5*(X(n,k)*X(k,m)+Y(n,k)*Y(k,m)))*&
                        sum=0.5*(X(n,k)*X(k,m))*&
                        &( (E(k)-E(m))*EXP(i*(E(n)-E(k))*t(t_index))&
                        & - (E(n)-E(k))*EXP(i*(E(k)-E(m))*t(t_index)) )
                        !$OMP critical
                        B(n,m,t_index)= B(n,m,t_index)+sum
                    end do
                end do
            end do 
        end do
        !$OMP END PARALLEL DO
    end subroutine get_b_omp


    subroutine calc_t_mat(t, xy,hbar,m,len_x,len_y,len_tot)!for DVR
        integer, intent(in) :: len_x, len_y, len_tot
        real(8), intent(in) :: hbar, m
        real(8), intent(in) :: xy(2,len_x, len_y)
        !f2py real(8), intent(in,out,copy) :: T(len_tot,len_tot)
        real(8), intent(out) :: T(len_tot, len_tot)
        integer :: cntr1,cntr2,i1,i2,j1,j2
        real(8) :: const_x,const_y,PI
        PI=4.D0*DATAN(1.D0)
        T(:,:)=0.0
        const_x = hbar**2.0 / (2.0*m* (xy(1,2,1)-xy(1,1,1))**2 )
        const_y = hbar**2.0 / (2.0*m* (xy(2,1,2)-xy(2,1,1))**2)
        do i1=1,len_x
            do j1=1,len_y
                cntr1=len_y*(i1-1)+j1
                do i2=1,len_x
                    do j2=1,len_y
                        cntr2=len_y*(i2-1)+j2
                        IF (i1==i2) THEN
                            T(cntr1,cntr2)=T(cntr1,cntr2)+const_y*PI**2/3
                        ELSE 
                            T(cntr1,cntr2)=T(cntr1,cntr2)+const_y*((-1)**(j1-j2) *2/((j1-j2)**2))
                        END IF
                        IF (j1==j2) THEN
                            T(cntr1,cntr2)=T(cntr1,cntr2)+const_x*PI**2/3
                        ELSE 
                            T(cntr1,cntr2)=T(cntr1,cntr2)+const_x*((-1)**(i1-i2) *2/((i1-i2)**2))
                        END IF
                    end do
                end do
            end do
        end do
    end subroutine calc_t_mat

    !for 2D: needs to be modified (only slightly though, maybe not at all) but Python is fast enough
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

end module otoc_2d


