// Program to digitize the MCU's internal temperature sensor's output voltage
// DriverLib Includes
#include <ti/devices/msp432p4xx/driverlib/driverlib.h>
// Standard Includes
#include <stdint.h>
#include <stdio.h>
#include <string.h>
volatile bool adc_done;
volatile uint32_t t = 0;
void delay(uint32_t delay_ms);
int main(void)
{
    // Halt WDT
    WDT_A_holdTimer();
    printf("MCLK: %u\n", MAP_CS_getMCLK());
    // Set reference voltage to 1.2 V and enable temperature sensor
    REF_A_enableReferenceVoltage();
    REF_A_enableTempSensor();
    REF_A_setReferenceVoltage(REF_A_VREF1_2V);
    // Initializing ADC (MCLK/1/1) with temperature sensor routed
    ADC14_enableModule();
    ADC14_initModule(ADC_CLOCKSOURCE_MCLK, ADC_PREDIVIDER_1, ADC_DIVIDER_1,ADC_TEMPSENSEMAP);
    // Set resolution
    ADC14_setResolution(ADC_14BIT);
    // Configure ADC Memory for temperature sensor data
    ADC14_configureSingleSampleMode(ADC_MEM0, false);
    ADC14_configureConversionMemory(ADC_MEM0, ADC_VREFPOS_INTBUF_VREFNEG_VSS,ADC_INPUT_A22, false);
    // Configure the sample/hold time
    ADC14_setSampleHoldTime(ADC_PULSE_WIDTH_192, ADC_PULSE_WIDTH_192);
    // Enable sample timer in manual iteration mode and interrupts
    ADC14_enableSampleTimer(ADC_MANUAL_ITERATION);
    // Enable conversion
    ADC14_enableConversion();
    // Enabling Interrupts
    ADC14_enableInterrupt(ADC_INT0);
    Interrupt_enableInterrupt(INT_ADC14);
    Interrupt_enableMaster();
    uint32_t mclk_freq = MAP_CS_getMCLK();
    SysTick_setPeriod(mclk_freq / 1000);
    SysTick_enableInterrupt();
    SysTick_enableModule();
    // Trigger conversion with software
    while (1)
    {
        // Trigger conversion with software
        adc_done = false;
        ADC14_toggleConversionTrigger();

        // Wait for conversion to complete
        while (!adc_done)
        {
            // Busy wait
        }

        // Get the ADC result
        uint32_t adc_value = ADC14_getResult(ADC_MEM0);

        // Calculate the corresponding voltage in millivolts
        float voltage = (adc_value * 1200.0) / 16384.0; // 1.2V reference, 14-bit resolution

        // Print the ADC value and voltage
        printf("ADC Value: %u, Voltage: %.2f mV\n", adc_value, voltage);

        // Add a delay of 500 ms
       delay(500);
    }
}

// This interrupt happens every time a conversion has completed
void ADC14_IRQHandler(void)
{
    uint64_t status;
    status = ADC14_getEnabledInterruptStatus();
    ADC14_clearInterruptFlag(status);
    if(status & ADC_INT0)
    {
        adc_done = true;
    }
}

void delay(uint32_t delay_ms)
{
    uint32_t start = t;

    while (t - start < delay_ms) ;
}
void SysTick_Handler(void)
{
    t++;
}