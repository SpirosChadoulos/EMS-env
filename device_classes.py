class Device:

    def __init__(self, name : str, device_power_consumption : float, user_probabilities : list, device_standard_penalty : float):

        '''
        This class is the parent class for all devices. It contains the basic attributes and methods that all devices have in common.

        Args:
        name : The name of the device
        device_power_consumption : The power consumption of the device in kilo watts
        user_probabilities : The ω, p, θ, q probabilities for the intermittent device
        device_standard_penalty : The penalty issued to the agent each time it proposes an action for this specific device that the user does not approve
        '''

        self.name = name
        self.device_power_consumption = device_power_consumption
        self.user_probabilities = user_probabilities
        self.device_standard_penalty = device_standard_penalty
        
    def get_name(self) -> str:

        '''
        This method returns the name of the device
        '''

        return self.name
    
    def get_device_power_consumption(self) -> float:

        '''
        This method returns the power consumption of the device
        '''

        return self.device_power_consumption
    
    def get_user_probabilities(self) -> list:

        '''
        This method returns the user probabilities of the device
        '''

        return self.user_probabilities
    
    def get_device_standard_penalty(self) -> float:

        '''
        This method returns the standard penalty of the device
        '''

        return self.device_standard_penalty


class Intermittent(Device):

    # Same as the parent class
    def __init__(self, name: str, device_power_consumption : float, user_probabilities : list, device_standard_penalty : float):

        '''
        This class is a child class of the Device class. It contains the attributes and methods that are specific to intermittent devices.
        '''

        super().__init__(name, device_power_consumption, user_probabilities, device_standard_penalty)

    

class Uninterruptible(Device):
    
    # Same as the parent class, but with some extra attributes
    def __init__(self, name: str, device_power_consumption : float, user_probabilities : list, device_standard_penalty : float, 
                 device_on_duration : float, device_override_penalty : float):

        '''
        This class is a child class of the Device class. It contains the attributes and methods that are specific to uninterruptible devices.

        Extra Args:
        device_on_duration : The duration the device has to stays on in order for its function to complete
        device_override_penalty : The penalty issued to the agent for overriding the device when it is still on use
        '''

        super().__init__(name, device_power_consumption, user_probabilities, device_standard_penalty)
        self.device_on_duration = device_on_duration
        self.device_override_penalty = device_override_penalty

    def get_device_on_duration(self) -> float:
            
            '''
            This method returns the duration the device has to stays on in order for its function to complete
            '''
    
            return self.device_on_duration
    
    def get_device_override_penalty(self) -> float:
            
            '''
            This method returns the penalty for overriding the device
            '''
    
            return self.device_override_penalty

def main():
    
    # Creating an intermittent device
    intermittent_device = Intermittent("Example Intermittent Device", 0.5, [0.5, 0.5, 0.5, 0.5], 0.5)
    print(intermittent_device.get_name())
    
    # Creating an uninterruptible device
    uninterruptible_device = Uninterruptible("Example Uninterruptible Device", 0.5, [0.5, 0.5, 0.5, 0.5], 0.5, 0.5, 0.5)
    print(uninterruptible_device.get_name())

if __name__ == "__main__":
    main()