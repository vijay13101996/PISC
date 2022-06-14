program string_array
    !use mymod, only : show, print_pairs_seq_ten
    use mymod
    implicit none
    character(len=10), dimension(2) :: keys, vals, other
    integer :: i
    ! 1D integer array
    integer, dimension(10) :: array1
    ! An equivalent array declaration
    integer :: array2(10)
    integer :: array3(10, 10)  ! 2D integer array of 100 elements

    keys = [character(len=10) :: "user", "dbname"]
    vals = [character(len=10) :: "ben", "motivation"]
    other = [character(len=10) :: "big", "ben"]
  
    call show(keys, vals)
    !call print_pairs_eq_ten()
    
  
  end program string_array

  module mymod
    implicit none 

    contains
  
    subroutine parallelizable_loop_sin()
        real, parameter :: pi = 3.14159265
        integer, parameter :: n = 10
        real :: result_sin(n)
        integer :: i

        do concurrent (i = 1:n)  ! Careful, the syntax is slightly different
            result_sin(i) = sin(i * pi/4.)
        end do

        print *, result_sin
    end subroutine parallelizable_loop_sin

    subroutine print_pairs_seq_ten()
        integer :: i, j
        outer_loop: do i = 1, 10
            inner_loop: do j = 1, 10
                if ((j + i) > 10) then  ! Print only pairs of i and j that add up to 10
                    cycle outer_loop  ! Go to the next iteration of the outer loop
                end if
                print *, 'I=', i, ' J=', j, ' Sum=', j + i
            end do inner_loop
        end do outer_loop
    end subroutine print_pairs_seq_ten

    subroutine show(akeys, avals)
      character(len=*), intent(in) :: akeys(:), avals(:) !intent(in)= read only , out: write only, inout: read and write
      integer                      :: i
  
      do i = 1, size(akeys)
        print *, trim(akeys(i)), ": ", trim(avals(i))
      end do
  
    end subroutine show

    subroutine print_odd_ints()
        integer :: i
        do i = 1, 10
            if (mod(i, 2) == 0) then
                cycle  ! Don't print even numbers
            end if
            print *, i
          end do
    end subroutine print_odd_ints
    
end module mymod
