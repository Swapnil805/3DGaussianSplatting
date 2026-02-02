using UnityEngine;

public class OrbitCamera : MonoBehaviour
{
    [Header("Target")]
    public Transform target;

    [Header("Distance")]
    public float distance = 5f;
    public float minDistance = 1.5f;
    public float maxDistance = 15f;

    [Header("Orbit")]
    public float orbitSpeed = 120f;
    public float minPitch = -20f;
    public float maxPitch = 80f;

    private float yaw = 0f;
    private float pitch = 20f;

    void Start()
    {
        if (target == null)
        {
            Debug.LogWarning("OrbitCamera: target not set.");
            return;
        }

        // Start from current camera position
        Vector3 offset = transform.position - target.position;
        distance = offset.magnitude;

        Vector3 angles = Quaternion.LookRotation(offset).eulerAngles;
        yaw = angles.y;
        pitch = Mathf.Clamp(angles.x, minPitch, maxPitch);
    }

    void LateUpdate()
    {
        if (target == null) return;

        // Right mouse drag = orbit
        if (Input.GetMouseButton(1))
        {
            float mouseX = Input.GetAxis("Mouse X");
            float mouseY = Input.GetAxis("Mouse Y");

            yaw += mouseX * orbitSpeed * Time.deltaTime;
            pitch -= mouseY * orbitSpeed * Time.deltaTime;
        }

        // Scroll wheel = zoom
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scroll) > 0.0001f)
        {
            distance -= scroll * (maxDistance - minDistance);
            distance = Mathf.Clamp(distance, minDistance, maxDistance);
        }

        // Apply rotation + position
        pitch = Mathf.Clamp(pitch, minPitch, maxPitch);
        Quaternion rotation = Quaternion.Euler(pitch, yaw, 0f);
        Vector3 offsetDir = rotation * Vector3.back; // (0,0,-1)

        Vector3 position = target.position + offsetDir * distance;
        transform.position = position;
        transform.LookAt(target);
    }
}
