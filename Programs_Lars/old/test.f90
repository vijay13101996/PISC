program test
    use otoc_1d
    implicit none
    integer :: N_trunc
    integer  :: N_grdpts
    real, allocatable :: grid(:) 
    real, allocatable :: EV(:,:)
    real,allocatable :: X(:,:)
    integer :: i !for the loop
    
    N_trunc = 6
    N_grdpts = 10

    allocate(grid(N_grdpts))
    allocate(X(N_trunc,N_trunc))
    allocate(EV(N_grdpts,N_trunc))
    grid = (/(i,i=1,N_grdpts, 1)/) !sample grid
    X(:,:) = 0 !initialize matrix el. to 0
    EV(:,:) = 1


    call fill_x_nm(X=X,EV=EV,grid=grid, N_grdpts=N_grdpts, N_trunc=N_trunc)
    !call test_2(X)
    !call which_version()! check program
end program test