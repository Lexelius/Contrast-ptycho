"""
Provides base and example classes for pseudo motors, which are
combinations of other motors.
"""

from . import Motor

class PseudoMotor(Motor):
    """
    Pseudo motor base class.
    """
    def __init__(self, motors, dry_run=False, *args, **kwargs):
        """
        Abstract base class for pseudo motors. The logic of pseudo
        motors needs some attention.

        :param motors: The underlying physical motors.
        :param dry_run: Don't move any physical motors, just print calculated positions.
        :type dry_run: bool
        """
        super(PseudoMotor, self).__init__(*args, **kwargs)
        self.motors = motors
        self.dry_run = dry_run

    def physicals(self):
        """
        Current positions of physical motors.

        :returns: Positions
        :rtype: list
        """
        return [m.position() for m in self.motors]

    @property
    def dial_position(self):
        return self.calc_pseudo(self.physicals())

    @dial_position.setter
    def dial_position(self, pos):
        physicals = self.calc_physicals(pos, self.physicals())
        for m, pos in zip(self.motors, physicals):
            if self.dry_run:
                print('Would move %s to %f' % (m.name, pos))
            else:
                m.move(pos)

    def busy(self):
        return True in [m.busy() for m in self.motors]

    def stop(self):
        [m.stop for m in self.motors()]

    def calc_pseudo(self, physicals):
        """
        Override this method, which calculates the pseudo position for
        given physical positions.

        :param physicals: Physical positions
        :returns: Pseudo position
        """
        raise NotImplementedError

    def calc_physicals(self, pseudo):
        """
        Override this method, which calculates the physical positions
        for a target pseudo position.

        :param pseudo: Target pseudo position
        :returns: Physical positions.
        """
        raise NotImplementedError

class ExamplePseudoMotor(PseudoMotor):
    """
    Example pseudo motor which implements the difference between two
    motors.
    """
    def calc_pseudo(self, physicals):
        return physicals[1] - physicals[0]

    def calc_physicals(self, pseudo):
        current_physicals = self.physicals
        current_diff = self.calc_pseudo(current_physicals)
        half_increase = (pseudo - current_diff) / 2.0
        return [current_physicals[0] - half_increase, 
                current_physicals[1] + half_increase]
